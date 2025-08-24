//! # Kachenjunga: Advanced Algorithms for Biological Quantum Computation
//! 
//! This library implements the complete S-Entropy Framework and biological quantum
//! computer algorithm ecosystem, providing the mathematical substrate for universal
//! problem solving through observer-process integration.
//! 
//! Developed by Kundai Farai Sachikonye under the divine protection of
//! Saint Stella-Lorraine Masunda, Patron Saint of Impossibility.

// Original modules
pub mod s_entropy;
pub mod bmd;
pub mod atp_dynamics;
pub mod oscillatory_systems;
pub mod temporal_navigation;
pub mod naked_thermodynamics;

// New complete algorithm suite
pub mod algorithms;

pub use s_entropy::*;
pub use bmd::*;
pub use atp_dynamics::*;
pub use algorithms::prelude::*;

/// Core mathematical constants used throughout the system
pub mod constants {
    /// The Saint Stella-Lorraine Masunda constant (STSL equation scaling)
    pub const STELLA_CONSTANT: f64 = 1.380649e-23;
    
    /// Divine intervention impossibility threshold
    pub const IMPOSSIBILITY_THRESHOLD: f64 = 1000.0;
    
    /// Atomic clock precision target (seconds)
    pub const ATOMIC_PRECISION_TARGET: f64 = 1e-12;
    
    /// Virtual Blood circulation efficiency target
    pub const VIRTUAL_BLOOD_EFFICIENCY: f64 = 0.997;
}
