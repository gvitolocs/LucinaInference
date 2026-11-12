//! Minimal GGUF v2/v3 reader (metadata + tensor directory over an mmap). Written from the format as
//! llama.cpp's `gguf.cpp` reads it; deliberately shares no code with the engine's loader.
use anyhow::{anyhow, bail, Context, Result};
use lucina_rules::gguf_bounds as gb;
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U(v) => Some(*v),
            Value::I(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F(v) => Some(*v),
            Value::U(v) => Some(*v as f64),
            Value::I(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the contiguous (row) dimension
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

pub struct Gguf {
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    map: memmap2::Mmap,
    data_start: usize,
    align: u64,
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        // proved: end <= len (gguf_bounds::take_end)
        let end = gb::take_end(self.p, n as u64, self.b.len())
            .ok_or_else(|| anyhow!("truncated gguf at byte {}", self.p))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn remaining(&self) -> u64 {
        self.b.len().saturating_sub(self.p) as u64
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if !gb::count_fits(n, 1, self.remaining()) {
            bail!("string of {n} bytes at byte {} runs past the file", self.p);
        }
        Ok(String::from_utf8_lossy(self.take(n as usize)?).into_owned())
    }
    fn value(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::U(self.take(1)?[0] as u64),
            1 => Value::I(self.take(1)?[0] as i8 as i64),
            2 => Value::U(u16::from_le_bytes(self.take(2)?.try_into()?) as u64),
            3 => Value::I(i16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(self.u32()? as i32 as i64),
            6 => Value::F(f32::from_le_bytes(self.take(4)?.try_into()?) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()?;
                // every element takes at least one byte: a count the file cannot hold is refused before allocating
                if !gb::count_fits(n, 1, self.remaining()) {
                    bail!(
                        "array of {n} elements at byte {} runs past the file",
                        self.p
                    );
                }
                let n = n as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Arr(v)
            }
            10 => Value::U(self.u64()?),
            11 => Value::I(self.u64()? as i64),
            12 => Value::F(f64::from_le_bytes(self.take(8)?.try_into()?)),
            _ => bail!("unknown gguf value type {t}"),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: read-only map of a model file nobody writes while we run.
        let map = unsafe { memmap2::Mmap::map(&f)? };
        let mut r = Rd { b: &map, p: 0 };
        if r.take(4)? != b"GGUF" {
            bail!("{}: not a GGUF file", path.display());
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;
        if !gb::count_fits(n_kv, gb::MIN_KV_BYTES, r.remaining()) {
            bail!("{n_kv} metadata pairs cannot fit in the file");
        }
        if !gb::count_fits(n_tensors, gb::MIN_TENSOR_INFO_BYTES, r.remaining()) {
            bail!("{n_tensors} tensors cannot fit in the file");
        }
        let mut kv = HashMap::new();
        for _ in 0..n_kv {
            let k = r.string()?;
            let t = r.u32()?;
            kv.insert(k, r.value(t)?);
        }
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()? as usize;
            if nd == 0 || nd > gb::MAX_DIMS {
                bail!(
                    "tensor {name}: {nd} dimensions (1..={} allowed)",
                    gb::MAX_DIMS
                );
            }
            let dims = (0..nd).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
            let ggml_type = r.u32()?;
            let offset = r.u64()?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dims,
                    ggml_type,
                    offset,
                },
            );
        }
        let align = kv
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);
        if align == 0 || !align.is_power_of_two() {
            bail!("general.alignment {align} is not a power of two");
        }
        let data_start = gb::align_up(r.p as u64, align)
            .ok_or_else(|| anyhow!("data section start overflows"))?;
        if data_start > map.len() as u64 {
            bail!("data section starts past the end of the file");
        }
        Ok(Self {
            kv,
            tensors,
            map,
            data_start: data_start as usize,
            align,
        })
    }

    pub fn u(&self, key: &str) -> Result<u64> {
        self.kv
            .get(key)
            .and_then(|v| v.as_u64())
            .with_context(|| format!("missing integer key {key}"))
    }
    pub fn f(&self, key: &str) -> Result<f64> {
        self.kv
            .get(key)
            .and_then(|v| v.as_f64())
            .with_context(|| format!("missing float key {key}"))
    }
    pub fn s(&self, key: &str) -> Result<&str> {
        self.kv
            .get(key)
            .and_then(|v| v.as_str())
            .with_context(|| format!("missing string key {key}"))
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }

    /// Raw bytes of a tensor.
    pub fn bytes(&self, t: &TensorInfo) -> Result<&[u8]> {
        // proved (gguf_bounds): no overflowing element count, whole blocks, aligned offset, range inside the file
        let numel = gb::numel(&t.dims)
            .with_context(|| format!("tensor {}: bad dimensions {:?}", t.name, t.dims))?;
        let (blk, sz) = crate::quant::block_layout(t.ggml_type)?;
        let (start, len) = gb::tensor_span(
            self.map.len() as u64,
            self.data_start as u64,
            t.offset,
            numel,
            blk as u64,
            sz as u64,
            self.align,
        )
        .with_context(|| format!("tensor {} out of file bounds or misaligned", t.name))?;
        Ok(&self.map[start as usize..(start + len) as usize])
    }
}
