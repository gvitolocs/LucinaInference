//! Minimal GGUF v2/v3 reader (metadata + tensor directory over an mmap). Written from the format as
//! llama.cpp's `gguf.cpp` reads it; deliberately shares no code with the engine's loader.
use anyhow::{bail, Context, Result};
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
    /// ggml order: dims[0] is the contiguous (row) dimension
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

pub struct Gguf {
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    map: memmap2::Mmap,
    data_start: usize,
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            bail!("truncated gguf at byte {}", self.p);
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
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
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
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
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;
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
            let dims = (0..nd).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
            let ggml_type = r.u32()?;
            let offset = r.u64()?;
            tensors.insert(name.clone(), TensorInfo { name, dims, ggml_type, offset });
        }
        let align = kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let data_start = r.p.div_ceil(align) * align;
        Ok(Self { kv, tensors, map, data_start })
    }

    pub fn u(&self, key: &str) -> Result<u64> {
        self.kv.get(key).and_then(|v| v.as_u64()).with_context(|| format!("missing integer key {key}"))
    }
    pub fn f(&self, key: &str) -> Result<f64> {
        self.kv.get(key).and_then(|v| v.as_f64()).with_context(|| format!("missing float key {key}"))
    }
    pub fn s(&self, key: &str) -> Result<&str> {
        self.kv.get(key).and_then(|v| v.as_str()).with_context(|| format!("missing string key {key}"))
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("missing tensor {name}"))
    }

    /// Raw bytes of a tensor.
    pub fn bytes(&self, t: &TensorInfo) -> Result<&[u8]> {
        let numel: u64 = t.dims.iter().product();
        let (blk, sz) = crate::quant::block_layout(t.ggml_type)?;
        let n = (numel as usize / blk) * sz;
        let s = self.data_start + t.offset as usize;
        self.map.get(s..s + n).with_context(|| format!("tensor {} out of file bounds", t.name))
    }
}
