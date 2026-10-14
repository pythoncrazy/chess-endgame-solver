//! Read an exact JSON i64 array and write one interoperable binary frame.
use std::io::Read;
fn main() -> Result<(), String> {
    let output = std::env::args()
        .nth(1)
        .ok_or("usage: tb_encode OUTPUT_FRAME < VALUES_JSON")?;
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let values: Vec<i64> = serde_json::from_str(&input).map_err(|e| e.to_string())?;
    let frame = tablebase_core::encode::encode(&values, 3)?;
    std::fs::write(output, frame).map_err(|e| e.to_string())?;
    Ok(())
}
