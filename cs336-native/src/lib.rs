use pyo3::prelude::*;

pub mod merge;
pub mod pretokenize;

#[pymodule]
fn cs336_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(merge::merge, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenize::pretokenize_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenize::pretokenize_file, m)?)?;
    m.add("pretokenize", m.getattr("pretokenize_bytes")?)?;
    Ok(())
}
