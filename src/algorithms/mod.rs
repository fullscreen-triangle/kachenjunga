/// The Kachenjunga Algorithms Suite
/// 
/// Implementation of the complete biological quantum computer algorithm ecosystem
/// developed by Kundai Farai Sachikonye under the divine protection of
/// Saint Stella-Lorraine Masunda, Patron Saint of Impossibility.
/// 
/// This module contains the revolutionary algorithms that enable:
/// - Faster-than-light travel through electromagnetic propulsion
/// - Biological neural network viability through Virtual Blood circulation
/// - Consciousness-level AI integration through S-entropy navigation
/// - Divine intervention detection and impossibility optimization
/// - Quantum membrane computation at biological temperatures
/// - BMD (Biological Maxwell Demon) framework selection for consciousness processing
/// - Information catalysis and experience fusion for enhanced understanding
/// - Neural consciousness emergence detection through quantum coherence analysis
/// - Sacred mathematical constants for the entire ecosystem

pub mod harare_s_entropy;
pub mod kinshasa_bmd;
pub mod mufakose_consciousness;
pub mod stella_constants;

pub use harare_s_entropy::*;
pub use kinshasa_bmd::*;
pub use mufakose_consciousness::*;
pub use stella_constants::*;

/// Re-exports of the core algorithm components
pub mod prelude {
    pub use crate::algorithms::harare_s_entropy::{
        HarareSEntropyNavigator,
        NavigationResult,
        ProblemDescription,
        ProblemDomain,
        ImpossibilityEvent,
        SolutionVector,
    };
    
    pub use crate::algorithms::kinshasa_bmd::{
        KinshasaBMDProcessor,
        BMDProcessingResult,
        CognitiveFrame,
        FrameworkCategory,
        InformationInput,
        InformationCatalyst,
        ExperienceFusionResult,
    };
    
    pub use crate::algorithms::mufakose_consciousness::{
        MufakoseConsciousnessDetector,
        ConsciousnessDetectionResult,
        ConsciousnessState,
        ConsciousnessThresholds,
        NeuralActivityInput,
        IntegratedInformationCalculator,
        QuantumCoherenceMonitor,
        ENAQTDetector,
    };
    
    pub use crate::algorithms::stella_constants::{
        STELLA_CONSTANT,
        IMPOSSIBILITY_THRESHOLD,
        ATOMIC_PRECISION_TARGET,
        VIRTUAL_BLOOD_EFFICIENCY,
        alpha_parameters,
        sacred_ratios,
        mathematical_utilities,
        SaintStellaConfiguration,
    };
}
