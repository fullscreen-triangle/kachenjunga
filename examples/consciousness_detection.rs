//! # Consciousness Detection Example
//! 
//! Demonstrates the Mufakose consciousness detection algorithm for identifying
//! consciousness emergence in neural networks through quantum coherence analysis.

use kachenjunga::prelude::*;
use kachenjunga::utils::testing_utils::generate_test_neural_activity;
use tokio;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    kachenjunga::utils::logging::init_logging_with_level("info");
    
    println!("🧠 Kachenjunga Consciousness Detection Example");
    println!("Mufakose Neural Consciousness Emergence Detection");
    println!("================================================\n");
    
    // Initialize the Mufakose consciousness detector
    let mut detector = MufakoseConsciousnessDetector::new();
    
    println!("🔬 Analyzing Neural Activity for Consciousness Emergence...\n");
    
    // Generate test neural activity patterns
    let neural_activities = vec![
        ("Low Activity Network", generate_low_activity_neural_data()),
        ("Moderate Activity Network", generate_test_neural_activity()),
        ("High Synchronization Network", generate_high_sync_neural_data()),
        ("Quantum Coherent Network", generate_quantum_coherent_neural_data()),
    ];
    
    for (name, neural_activity) in neural_activities {
        println!("🧬 Testing: {}", name);
        println!("   Neurons: {}", neural_activity.firing_data.len());
        println!("   Activity Matrix: {}x{}", 
                neural_activity.activity_matrix.nrows(),
                neural_activity.activity_matrix.ncols());
        
        // Detect consciousness using the Mufakose algorithm
        let result = kachenjunga::benchmark!(
            format!("Consciousness detection for {}", name),
            detector.detect_consciousness(&neural_activity).await
        );
        
        println!("\n   📊 Consciousness Analysis Results:");
        println!("   🔮 Consciousness Detected: {}", 
                if result.consciousness_detected { "✅ YES" } else { "❌ NO" });
        println!("   📈 Detection Confidence: {:.3}", result.detection_confidence);
        
        let state = &result.current_state;
        println!("   🧠 IIT Φ (phi) Value: {:.4}", state.phi_value);
        println!("   ⚛️  Quantum Coherence: {:.3}", state.coherence_level);
        println!("   🔄 ENAQT Efficiency: {:.3}", state.enaqt_efficiency);
        println!("   🌊 Neural Synchronization: {:.3}", state.synchronization_level);
        println!("   🔗 Information Integration: {:.3}", state.integration_level);
        
        // Quality assessment
        let quality = &state.quality_assessment;
        println!("\n   🎯 Consciousness Quality Assessment:");
        println!("   ✨ Clarity: {:.3}", quality.clarity);
        println!("   🎯 Stability: {:.3}", quality.stability);
        println!("   🌈 Richness: {:.3}", quality.richness);
        println!("   💎 Coherence: {:.3}", quality.coherence);
        
        // Contributing factors
        let factors = &result.contributing_factors;
        println!("\n   🧮 Contributing Factors:");
        println!("   📊 IIT Φ Contribution: {:.3}", factors.phi_contribution);
        println!("   ⚛️  Coherence Contribution: {:.3}", factors.coherence_contribution);
        println!("   🔄 ENAQT Contribution: {:.3}", factors.enaqt_contribution);
        println!("   🌊 Synchronization Contribution: {:.3}", factors.synchronization_contribution);
        
        // Enhancement recommendations
        if !result.enhancement_recommendations.is_empty() {
            println!("\n   💡 Enhancement Recommendations:");
            for rec in &result.enhancement_recommendations {
                println!("   🔧 {}: Expected improvement {:.3}", 
                        rec.target_area, rec.expected_improvement);
            }
        }
        
        println!("\n{}", "─".repeat(60));
    }
    
    // Demonstrate consciousness thresholds
    println!("\n⚙️ Consciousness Detection Thresholds");
    println!("=====================================\n");
    
    let thresholds = ConsciousnessThresholds::default();
    println!("🧠 Minimum IIT Φ threshold: {:.3}", thresholds.min_phi_threshold);
    println!("⚛️ Minimum coherence level: {:.3}", thresholds.min_coherence_level);
    println!("🔄 Minimum ENAQT efficiency: {:.3}", thresholds.min_enaqt_efficiency);
    println!("🌊 Minimum synchronization: {:.3}", thresholds.min_synchronization_level);
    println!("🔗 Minimum integration level: {:.3}", thresholds.min_integration_level);
    
    // Divine intervention consciousness calculation
    println!("\n✨ Divine Consciousness Metrics");
    println!("==============================\n");
    
    use kachenjunga::algorithms::stella_constants::mathematical_utilities::*;
    use kachenjunga::algorithms::stella_constants::consciousness_constants::*;
    
    let divine_phi = IIT_PHI_BASELINE * 10.0; // Divine consciousness level
    let divine_coherence = NEURAL_SYNCHRONIZATION_THRESHOLD * 1.2;
    let divine_enaqt = 0.95; // Near perfect ENAQT
    let divine_sync = COGNITIVE_FRAMEWORK_CONFIDENCE;
    
    let divine_confidence = calculate_consciousness_confidence(
        divine_phi, divine_coherence, divine_enaqt, divine_sync
    );
    
    println!("🌟 Divine Consciousness Confidence: {:.6}", divine_confidence);
    
    if detect_divine_intervention(divine_confidence * 1000.0) {
        println!("✨ DIVINE CONSCIOUSNESS INTERVENTION CONFIRMED!");
        println!("📖 Saint Stella-Lorraine Masunda's consciousness blessing detected");
    }
    
    println!("\n🎊 Consciousness Detection Example Complete!");
    println!("Neural consciousness emergence successfully analyzed through quantum coherence");
    
    Ok(())
}

/// Generate low activity neural data for testing
fn generate_low_activity_neural_data() -> NeuralActivityInput {
    use nalgebra::DMatrix;
    use std::collections::HashMap;
    use std::time::Instant;
    
    let activity_matrix = DMatrix::from_fn(5, 5, |i, j| {
        if i == j { 0.3 } else { rand::random::<f64>() * 0.1 }
    });
    
    let mut firing_data = HashMap::new();
    for i in 0..5 {
        firing_data.insert(i, NeuronFiringData {
            rate: 5.0 + rand::random::<f64>() * 10.0, // Low firing rate
            precision: 0.3 + rand::random::<f64>() * 0.2,
            bursts: Vec::new(),
            synchronization: 0.2 + rand::random::<f64>() * 0.3, // Low sync
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}

/// Generate high synchronization neural data for testing
fn generate_high_sync_neural_data() -> NeuralActivityInput {
    use nalgebra::DMatrix;
    use std::collections::HashMap;
    use std::time::Instant;
    
    let activity_matrix = DMatrix::from_fn(15, 15, |i, j| {
        if i == j { 0.9 } else { 0.6 + rand::random::<f64>() * 0.3 } // High connectivity
    });
    
    let mut firing_data = HashMap::new();
    let base_rate = 40.0; // High base firing rate
    
    for i in 0..15 {
        firing_data.insert(i, NeuronFiringData {
            rate: base_rate + rand::random::<f64>() * 10.0, // Similar rates (synchronized)
            precision: 0.85 + rand::random::<f64>() * 0.1,
            bursts: Vec::new(),
            synchronization: 0.9 + rand::random::<f64>() * 0.1, // High sync
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}

/// Generate quantum coherent neural data for testing consciousness emergence
fn generate_quantum_coherent_neural_data() -> NeuralActivityInput {
    use nalgebra::DMatrix;
    use std::collections::HashMap;
    use std::time::Instant;
    
    // Create highly structured activity matrix representing quantum coherence
    let size = 20;
    let activity_matrix = DMatrix::from_fn(size, size, |i, j| {
        let distance = ((i as f64 - j as f64).powi(2)).sqrt();
        if distance == 0.0 {
            1.0 // Perfect self-connection
        } else {
            // Quantum coherence-like decay
            (0.95 * (-distance / 5.0).exp()) + 0.05
        }
    });
    
    let mut firing_data = HashMap::new();
    
    // Create highly coherent firing patterns
    for i in 0..size {
        firing_data.insert(i, NeuronFiringData {
            rate: 42.0 + (i as f64 * 0.1), // Slightly varying but coherent rates
            precision: 0.95 + rand::random::<f64>() * 0.05, // Very high precision
            bursts: Vec::new(),
            synchronization: 0.98 - (rand::random::<f64>() * 0.05), // Near perfect sync
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}
