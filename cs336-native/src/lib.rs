use pyo3::prelude::*;

pub mod merge;

#[pymodule]
fn cs336_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(merge::merge, m)?)?;
    Ok(())
}
