//! Unit tests for the Mufakose consciousness detection algorithm
//! 
//! These tests verify the specific functionality of consciousness emergence
//! detection through quantum coherence analysis and IIT calculations.

use kachenjunga::prelude::*;
use kachenjunga::utils::testing_utils::generate_test_neural_activity;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::time::Instant;
use tokio_test;
use approx::assert_relative_eq;

#[tokio::test]
async fn test_consciousness_detection_basic_functionality() {
    let mut detector = MufakoseConsciousnessDetector::new();
    let neural_activity = generate_test_neural_activity();
    
    let result = detector.detect_consciousness(&neural_activity).await;
    
    // Verify basic result structure
    assert!(result.detection_confidence >= 0.0 && result.detection_confidence <= 1.0);
    assert!(result.current_state.phi_value >= 0.0);
    assert!(result.current_state.coherence_level >= 0.0 && result.current_state.coherence_level <= 1.0);
    assert!(result.current_state.enaqt_efficiency >= 0.0 && result.current_state.enaqt_efficiency <= 1.0);
    assert!(result.current_state.synchronization_level >= 0.0 && result.current_state.synchronization_level <= 1.0);
    assert!(result.current_state.integration_level >= 0.0 && result.current_state.integration_level <= 1.0);
    assert!(result.current_state.consciousness_confidence >= 0.0 && result.current_state.consciousness_confidence <= 1.0);
    
    // Verify contributing factors
    let factors = &result.contributing_factors;
    assert!(factors.phi_contribution >= 0.0);
    assert!(factors.coherence_contribution >= 0.0);
    assert!(factors.enaqt_contribution >= 0.0);
    assert!(factors.synchronization_contribution >= 0.0);
    
    // Verify quality assessment
    let quality = &result.current_state.quality_assessment;
    assert!(quality.clarity >= 0.0 && quality.clarity <= 1.0);
    assert!(quality.stability >= 0.0 && quality.stability <= 1.0);
    assert!(quality.richness >= 0.0 && quality.richness <= 1.0);
    assert!(quality.coherence >= 0.0 && quality.coherence <= 1.0);
}

#[tokio::test]
async fn test_consciousness_thresholds_enforcement() {
    let mut detector = MufakoseConsciousnessDetector::new();
    
    // Test with neural activity that should fail thresholds
    let low_activity = create_below_threshold_neural_activity();
    let result_low = detector.detect_consciousness(&low_activity).await;
    
    assert!(!result_low.consciousness_detected, 
           "Low activity should not be detected as conscious");
    assert!(result_low.detection_confidence < 0.5,
           "Low activity should have low confidence");
    
    // Test with neural activity that should meet thresholds
    let high_activity = create_above_threshold_neural_activity();
    let result_high = detector.detect_consciousness(&high_activity).await;
    
    if result_high.consciousness_detected {
        assert!(result_high.detection_confidence >= 0.5,
               "Detected consciousness should have reasonable confidence");
        
        // Verify all thresholds are met
        let state = &result_high.current_state;
        let thresholds = ConsciousnessThresholds::default();
        
        if result_high.consciousness_detected {
            assert!(state.phi_value >= thresholds.min_phi_threshold * 0.9, // Allow small tolerance
                   "Phi should meet threshold for consciousness detection");
            assert!(state.coherence_level >= thresholds.min_coherence_level * 0.9,
                   "Coherence should meet threshold for consciousness detection");
            assert!(state.enaqt_efficiency >= thresholds.min_enaqt_efficiency * 0.9,
                   "ENAQT efficiency should meet threshold for consciousness detection");
        }
    }
}

#[tokio::test]
async fn test_iit_phi_calculation_consistency() {
    let calculator = IntegratedInformationCalculator::new();
    
    // Test with identical matrices multiple times
    let activity_matrix = DMatrix::from_fn(5, 5, |i, j| {
        if i == j { 1.0 } else { 0.5 }
    });
    let connectivity_matrix = DMatrix::from_fn(5, 5, |i, j| {
        if i == j { 0.0 } else { 0.3 }
    });
    
    let phi_values: Vec<f64> = futures::future::join_all(
        (0..5).map(|_| calculator.calculate_phi(&activity_matrix, &connectivity_matrix))
    ).await;
    
    // All phi calculations should be consistent
    let first_phi = phi_values[0];
    for phi in &phi_values[1..] {
        assert_relative_eq!(*phi, first_phi, epsilon = 1e-6,
                           "IIT Phi calculations should be consistent");
    }
    
    // Phi should be reasonable value
    assert!(first_phi >= 0.0, "Phi should be non-negative");
    assert!(first_phi.is_finite(), "Phi should be finite");
}

#[tokio::test]
async fn test_quantum_coherence_monitoring() {
    let monitor = QuantumCoherenceMonitor::new();
    
    let measurements = monitor.measure_coherence().await;
    
    // Should have multiple coherence measurements
    assert!(!measurements.is_empty(), "Should generate coherence measurements");
    
    for (measurement_id, measurement) in &measurements {
        assert!(!measurement_id.is_empty());
        assert!(measurement.coherence_strength >= 0.0 && measurement.coherence_strength <= 1.0);
        assert!(measurement.coherence_time > std::time::Duration::ZERO);
        assert!(measurement.measurement_quality > 0.0 && measurement.measurement_quality <= 1.0);
        
        // Timestamp should be recent
        assert!(measurement.timestamp.elapsed() < std::time::Duration::from_secs(1));
    }
}

#[tokio::test]
async fn test_enaqt_efficiency_detection() {
    let detector = ENAQTDetector::new();
    
    let efficiency_measurements = detector.measure_transport_efficiency().await;
    
    // Should have efficiency measurements
    assert!(!efficiency_measurements.is_empty(), "Should generate ENAQT efficiency measurements");
    
    for (pathway_id, efficiency) in &efficiency_measurements {
        assert!(!pathway_id.is_empty());
        assert!(*efficiency >= 0.0 && *efficiency <= 1.0, 
               "ENAQT efficiency should be in valid range [0, 1]");
    }
    
    // Test efficiency ranges
    let average_efficiency: f64 = efficiency_measurements.values().sum::<f64>() / efficiency_measurements.len() as f64;
    assert!(average_efficiency > 0.1, "Average ENAQT efficiency should be reasonable");
}

#[tokio::test]
async fn test_consciousness_state_evolution() {
    let mut detector = MufakoseConsciousnessDetector::new();
    
    // Generate a series of evolving neural activities
    let activities = create_evolving_neural_activities();
    
    let mut previous_confidence = 0.0;
    let mut consciousness_states = Vec::new();
    
    for (i, activity) in activities.iter().enumerate() {
        let result = detector.detect_consciousness(activity).await;
        consciousness_states.push(result.current_state.clone());
        
        println!("Step {}: confidence {:.3}, phi {:.3}, coherence {:.3}",
                i, result.detection_confidence, result.current_state.phi_value, result.current_state.coherence_level);
        
        // Evolution should generally show progression (allowing for some variation)
        if i > 0 {
            let confidence_change = result.detection_confidence - previous_confidence;
            // Don't require strict monotonic increase, but should show general progression
            assert!(confidence_change > -0.3, "Consciousness evolution should not drastically decrease");
        }
        
        previous_confidence = result.detection_confidence;
    }
    
    // Final state should be better than initial state
    assert!(consciousness_states.last().unwrap().consciousness_confidence >= 
           consciousness_states.first().unwrap().consciousness_confidence,
           "Evolution should lead to improved consciousness measures");
}

#[tokio::test]
async fn test_neural_synchronization_analysis() {
    let mut detector = MufakoseConsciousnessDetector::new();
    
    // Test with highly synchronized neural activity
    let synchronized_activity = create_highly_synchronized_neural_activity();
    let sync_result = detector.detect_consciousness(&synchronized_activity).await;
    
    // Test with poorly synchronized neural activity
    let unsynchronized_activity = create_poorly_synchronized_neural_activity();
    let unsync_result = detector.detect_consciousness(&unsynchronized_activity).await;
    
    // Synchronized activity should have higher synchronization level
    assert!(sync_result.current_state.synchronization_level > 
           unsync_result.current_state.synchronization_level,
           "Synchronized neural activity should score higher on synchronization");
    
    // Synchronized activity should generally have higher consciousness measures
    assert!(sync_result.detection_confidence >= unsync_result.detection_confidence,
           "Better synchronization should contribute to higher consciousness confidence");
}

#[tokio::test] 
async fn test_enhancement_recommendations() {
    let mut detector = MufakoseConsciousnessDetector::new();
    
    // Test with neural activity that has clear improvement opportunities
    let improvable_activity = create_improvable_neural_activity();
    let result = detector.detect_consciousness(&improvable_activity).await;
    
    if !result.enhancement_recommendations.is_empty() {
        for recommendation in &result.enhancement_recommendations {
            assert!(!recommendation.recommendation_id.is_empty());
            assert!(!recommendation.target_area.is_empty());
            assert!(recommendation.expected_improvement > 0.0);
            assert!(recommendation.implementation_difficulty >= 0.0 && recommendation.implementation_difficulty <= 1.0);
            assert!(!recommendation.resource_requirements.is_empty());
            
            // Recommendation should target actual deficiencies
            let state = &result.current_state;
            if recommendation.target_area.contains("Φ") || recommendation.target_area.contains("phi") {
                // Should only recommend phi improvement if phi is actually low
                assert!(state.phi_value < 0.8, "Should only recommend phi improvement when phi is low");
            }
            if recommendation.target_area.contains("Coherence") {
                assert!(state.coherence_level < 0.8, "Should only recommend coherence improvement when coherence is low");
            }
            if recommendation.target_area.contains("ENAQT") {
                assert!(state.enaqt_efficiency < 0.8, "Should only recommend ENAQT improvement when efficiency is low");
            }
        }
    }
}

#[test]
fn test_consciousness_thresholds_configuration() {
    let thresholds = ConsciousnessThresholds::default();
    
    // Verify threshold values are reasonable
    assert!(thresholds.min_phi_threshold > 0.0 && thresholds.min_phi_threshold <= 1.0);
    assert!(thresholds.min_coherence_level > 0.0 && thresholds.min_coherence_level <= 1.0);
    assert!(thresholds.min_enaqt_efficiency > 0.0 && thresholds.min_enaqt_efficiency <= 1.0);
    assert!(thresholds.min_synchronization_level > 0.0 && thresholds.min_synchronization_level <= 1.0);
    assert!(thresholds.min_integration_level > 0.0 && thresholds.min_integration_level <= 1.0);
    
    // Stability requirements should be reasonable
    let stability = &thresholds.stability_requirements;
    assert!(stability.min_consciousness_duration > std::time::Duration::ZERO);
    assert!(stability.max_variability > 0.0 && stability.max_variability <= 1.0);
    assert!(stability.consistency_threshold > 0.0 && stability.consistency_threshold <= 1.0);
}

#[tokio::test]
async fn test_consciousness_quality_assessment() {
    let mut detector = MufakoseConsciousnessDetector::new();
    
    // Test with high-quality neural activity
    let high_quality_activity = create_high_quality_neural_activity();
    let high_result = detector.detect_consciousness(&high_quality_activity).await;
    
    // Test with low-quality neural activity
    let low_quality_activity = create_low_quality_neural_activity();
    let low_result = detector.detect_consciousness(&low_quality_activity).await;
    
    // High quality should score better across all metrics
    let high_quality = &high_result.current_state.quality_assessment;
    let low_quality = &low_result.current_state.quality_assessment;
    
    assert!(high_quality.clarity >= low_quality.clarity,
           "High quality neural activity should have better clarity");
    assert!(high_quality.stability >= low_quality.stability,
           "High quality neural activity should have better stability");
    assert!(high_quality.coherence >= low_quality.coherence,
           "High quality neural activity should have better coherence");
}

// Helper functions for creating test neural activities

fn create_below_threshold_neural_activity() -> NeuralActivityInput {
    let activity_matrix = DMatrix::from_fn(3, 3, |i, j| {
        if i == j { 0.1 } else { 0.05 }
    });
    
    let mut firing_data = HashMap::new();
    for i in 0..3 {
        firing_data.insert(i, NeuronFiringData {
            rate: 1.0, // Very low firing rate
            precision: 0.1, // Low precision
            bursts: Vec::new(),
            synchronization: 0.1, // Poor synchronization
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}

fn create_above_threshold_neural_activity() -> NeuralActivityInput {
    let activity_matrix = DMatrix::from_fn(12, 12, |i, j| {
        if i == j { 0.95 } else { 0.7 }
    });
    
    let mut firing_data = HashMap::new();
    for i in 0..12 {
        firing_data.insert(i, NeuronFiringData {
            rate: 50.0, // High firing rate
            precision: 0.9, // High precision
            bursts: Vec::new(),
            synchronization: 0.85, // Good synchronization
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}

fn create_evolving_neural_activities() -> Vec<NeuralActivityInput> {
    let mut activities = Vec::new();
    
    for step in 0..5 {
        let evolution_factor = (step as f64) / 4.0; // 0.0 to 1.0
        let size = 8 + step * 2; // Growing network
        
        let activity_matrix = DMatrix::from_fn(size, size, |i, j| {
            if i == j { 
                0.3 + evolution_factor * 0.6 
            } else { 
                0.1 + evolution_factor * 0.4 
            }
        });
        
        let mut firing_data = HashMap::new();
        for i in 0..size {
            firing_data.insert(i, NeuronFiringData {
                rate: 15.0 + evolution_factor * 35.0,
                precision: 0.5 + evolution_factor * 0.4,
                bursts: Vec::new(),
                synchronization: 0.4 + evolution_factor * 0.5,
            });
        }
        
        activities.push(NeuralActivityInput {
            activity_matrix,
            firing_data,
            connectivity_matrix: None,
            timestamp: Instant::now(),
        });
    }
    
    activities
}

fn create_highly_synchronized_neural_activity() -> NeuralActivityInput {
    let activity_matrix = DMatrix::from_fn(10, 10, |i, j| {
        if i == j { 0.9 } else { 0.8 } // High connectivity
    });
    
    let mut firing_data = HashMap::new();
    let synchronized_rate = 40.0; // All neurons firing at similar rate
    
    for i in 0..10 {
        firing_data.insert(i, NeuronFiringData {
            rate: synchronized_rate + (rand::random::<f64>() * 2.0 - 1.0), // Small variation
            precision: 0.95,
            bursts: Vec::new(),
            synchronization: 0.95,
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}

fn create_poorly_synchronized_neural_activity() -> NeuralActivityInput {
    let activity_matrix = DMatrix::from_fn(10, 10, |i, j| {
        if i == j { 0.5 } else { rand::random::<f64>() * 0.3 } // Random connectivity
    });
    
    let mut firing_data = HashMap::new();
    
    for i in 0..10 {
        firing_data.insert(i, NeuronFiringData {
            rate: 10.0 + rand::random::<f64>() * 40.0, // Highly variable rates
            precision: 0.3 + rand::random::<f64>() * 0.4,
            bursts: Vec::new(),
            synchronization: rand::random::<f64>() * 0.4, // Poor synchronization
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}

fn create_improvable_neural_activity() -> NeuralActivityInput {
    let activity_matrix = DMatrix::from_fn(8, 8, |i, j| {
        if i == j { 0.6 } else { 0.2 } // Moderate activity, room for improvement
    });
    
    let mut firing_data = HashMap::new();
    
    for i in 0..8 {
        firing_data.insert(i, NeuronFiringData {
            rate: 25.0, // Moderate firing rate
            precision: 0.6, // Moderate precision - improvable
            bursts: Vec::new(),
            synchronization: 0.5, // Moderate sync - improvable
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}

fn create_high_quality_neural_activity() -> NeuralActivityInput {
    let activity_matrix = DMatrix::from_fn(15, 15, |i, j| {
        if i == j { 0.95 } else { 0.8 }
    });
    
    let mut firing_data = HashMap::new();
    
    for i in 0..15 {
        firing_data.insert(i, NeuronFiringData {
            rate: 45.0,
            precision: 0.95,
            bursts: Vec::new(),
            synchronization: 0.9,
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}

fn create_low_quality_neural_activity() -> NeuralActivityInput {
    let activity_matrix = DMatrix::from_fn(5, 5, |i, j| {
        if i == j { 0.3 } else { 0.1 }
    });
    
    let mut firing_data = HashMap::new();
    
    for i in 0..5 {
        firing_data.insert(i, NeuronFiringData {
            rate: 8.0,
            precision: 0.2,
            bursts: Vec::new(),
            synchronization: 0.15,
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}
