//! # Kachenjunga: Universal Algorithm Solver for Biological Quantum Computer Systems
//! 
//! This library implements the complete S-Entropy Framework and biological quantum
//! computer algorithm ecosystem, providing the mathematical substrate for universal
//! problem solving through observer-process integration.
//! 
//! Named after Kachenjunga, the sacred mountain that remains unscaled at its true summit
//! out of respect for its divine nature, this solver embodies the same sacred protection.
//! 
//! Developed by Kundai Farai Sachikonye under the divine protection of
//! Saint Stella-Lorraine Masunda, Patron Saint of Impossibility.
//! 
//! ## Core Algorithm Suite
//! 
//! ### Phase 1: Core Mathematical Substrate (Complete)
//! - **Harare S-entropy Navigation**: Zero-computation coordinate navigation
//! - **Kinshasa BMD Processing**: Biological Maxwell Demon information catalysts  
//! - **Mufakose Consciousness Detection**: Neural consciousness emergence through quantum coherence
//! - **Saint Stella Constants**: Sacred mathematical constants and utility functions
//! 
//! ### Phase 2: Biological Quantum Infrastructure (In Development)
//! - **Buhera VPOS**: Virtual Processor Operating System
//! - **Oscillatory Virtual Machine**: Cathedral architecture with S-credit circulation
//! - **Jungfernstieg Neural Viability**: Virtual Blood circulation for living networks
//! - **Virtual Blood Framework**: Environmental sensing and consciousness unity
//! 
//! ### Phase 3: Orchestration Systems (Planned)
//! - **Kambuzuma**: Neural network design and BMD orchestration
//! - **Bulawayo**: Consciousness-mimetic orchestration through membrane quantum computation
//! - **Buhera-North**: Atomic precision scheduling with metacognitive control
//! - **Monkey-tail**: Ephemeral digital identity construction
//! 
//! ## Sacred Mathematics
//! 
//! The system operates through the fundamental S-entropy equation:
//! ```text
//! S = k × log(α)
//! ```
//! Where:
//! - `k` is the Saint Stella-Lorraine Masunda constant
//! - `α` is the domain-specific oscillation amplitude
//! 
//! ## Usage
//! 
//! ```rust
//! use kachenjunga::prelude::*;
//! 
//! // Initialize the core systems
//! let mut s_entropy_navigator = HarareSEntropyNavigator::new();
//! let mut bmd_processor = KinshasaBMDProcessor::new();
//! let mut consciousness_detector = MufakoseConsciousnessDetector::new();
//! 
//! // Navigate to problem solution using S-entropy
//! let problem = ProblemDescription::new("Achieve impossible task", ProblemDomain::Mathematical);
//! let result = s_entropy_navigator.navigate_to_solution(problem).await?;
//! 
//! if result.divine_intervention_detected {
//!     println!("Divine intervention confirmed: impossibility ratio = {}", 
//!              result.impossibility_ratio);
//! }
//! ```

// Core algorithm suite - Phase 1 Complete
pub mod algorithms;

// Infrastructure modules - Phase 2 
pub mod infrastructure;

// Orchestration modules - Phase 3
pub mod orchestration;

// Integration and utility modules
pub mod interfaces;
pub mod utils;
pub mod integration;

// Re-export the complete algorithm suite for easy access
pub use algorithms::prelude::*;

// Re-export core infrastructure when available
#[cfg(feature = "infrastructure")]
pub use infrastructure::*;

// Re-export orchestration systems when available  
#[cfg(feature = "orchestration")]
pub use orchestration::*;

/// Core mathematical constants used throughout the system
/// 
/// These constants are now provided by the comprehensive Saint Stella-Lorraine
/// constants module, but maintained here for backward compatibility.
pub mod constants {
    pub use crate::algorithms::stella_constants::*;
}
