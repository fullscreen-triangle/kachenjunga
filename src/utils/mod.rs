//! # Kachenjunga Utilities - Helper Functions and Tools
//! 
//! Utility functions and helper tools supporting the biological quantum computer
//! ecosystem across all phases of development.
//! 
//! ## Utility Components
//! 
//! - **Mathematical Utilities**: Mathematical helper functions and computations
//! - **Serialization**: Data serialization and deserialization support
//! - **Logging**: Structured logging and tracing utilities
//! - **Testing**: Testing utilities and helper functions
//! - **Benchmarking**: Performance benchmarking and measurement tools

pub mod mathematical_utils;
pub mod serialization;
pub mod logging;
pub mod testing_utils;
pub mod benchmarking;

// Re-export commonly used utilities
pub use mathematical_utils::*;
pub use serialization::*;
pub use logging::*;

// Mathematical utilities for the ecosystem
pub mod mathematical_utils {
    //! Mathematical utility functions supporting the Saint Stella-Lorraine framework
    pub use crate::algorithms::stella_constants::mathematical_utilities::*;
    
    /// Additional mathematical utilities will be implemented as needed
    /// during the development of subsequent phases.
}

// Serialization utilities
pub mod serialization {
    //! Serialization support for biological quantum computer data structures
    pub use serde::{Serialize, Deserialize};
    pub use serde_json;
    pub use bincode;
    
    /// Serialize any serializable data structure to JSON
    pub fn to_json<T: serde::Serialize>(data: &T) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(data)
    }
    
    /// Deserialize data from JSON
    pub fn from_json<T: for<'de> serde::Deserialize<'de>>(json: &str) -> Result<T, serde_json::Error> {
        serde_json::from_str(json)
    }
}

// Logging utilities
pub mod logging {
    //! Structured logging for the biological quantum computer ecosystem
    pub use tracing::{debug, info, warn, error, trace};
    pub use tracing_subscriber;
    
    /// Initialize logging with default configuration
    pub fn init_logging() {
        tracing_subscriber::fmt()
            .with_env_filter("info")
            .init();
    }
    
    /// Initialize logging with custom log level
    pub fn init_logging_with_level(level: &str) {
        tracing_subscriber::fmt()
            .with_env_filter(level)
            .init();
    }
}

// Testing utilities
pub mod testing_utils {
    //! Testing utilities for biological quantum computer components
    
    /// Generate test neural activity data for consciousness detection testing
    pub fn generate_test_neural_activity() -> crate::algorithms::mufakose_consciousness::NeuralActivityInput {
        use nalgebra::DMatrix;
        use std::collections::HashMap;
        use std::time::Instant;
        
        let activity_matrix = DMatrix::from_fn(10, 10, |i, j| {
            if i == j { 1.0 } else { rand::random::<f64>() * 0.5 }
        });
        
        let mut firing_data = HashMap::new();
        for i in 0..10 {
            firing_data.insert(i, crate::algorithms::mufakose_consciousness::NeuronFiringData {
                rate: 20.0 + rand::random::<f64>() * 30.0,
                precision: 0.8 + rand::random::<f64>() * 0.2,
                bursts: Vec::new(),
                synchronization: 0.7 + rand::random::<f64>() * 0.3,
            });
        }
        
        crate::algorithms::mufakose_consciousness::NeuralActivityInput {
            activity_matrix,
            firing_data,
            connectivity_matrix: None,
            timestamp: Instant::now(),
        }
    }
    
    /// Generate test problem descriptions for S-entropy navigation testing
    pub fn generate_test_problems() -> Vec<crate::algorithms::harare_s_entropy::ProblemDescription> {
        vec![
            crate::algorithms::harare_s_entropy::ProblemDescription::new(
                "Test mathematical problem".to_string(),
                crate::algorithms::harare_s_entropy::ProblemDomain::Mathematical
            ),
            crate::algorithms::harare_s_entropy::ProblemDescription::new(
                "Test biological optimization".to_string(), 
                crate::algorithms::harare_s_entropy::ProblemDomain::Biological
            ),
            crate::algorithms::harare_s_entropy::ProblemDescription::new(
                "Test consciousness integration".to_string(),
                crate::algorithms::harare_s_entropy::ProblemDomain::Consciousness
            ),
        ]
    }
}

// Benchmarking utilities
pub mod benchmarking {
    //! Performance benchmarking utilities for system optimization
    use std::time::{Duration, Instant};
    
    /// Simple benchmark timer
    pub struct BenchmarkTimer {
        start_time: Instant,
        description: String,
    }
    
    impl BenchmarkTimer {
        /// Create a new benchmark timer with description
        pub fn new(description: String) -> Self {
            Self {
                start_time: Instant::now(),
                description,
            }
        }
        
        /// Stop the timer and return elapsed duration
        pub fn stop(self) -> Duration {
            let elapsed = self.start_time.elapsed();
            crate::utils::logging::info!(
                "Benchmark '{}' completed in {:?}",
                self.description,
                elapsed
            );
            elapsed
        }
    }
    
    /// Macro for easy benchmarking
    #[macro_export]
    macro_rules! benchmark {
        ($description:expr, $code:expr) => {{
            let timer = crate::utils::benchmarking::BenchmarkTimer::new($description.to_string());
            let result = $code;
            timer.stop();
            result
        }};
    }
}
