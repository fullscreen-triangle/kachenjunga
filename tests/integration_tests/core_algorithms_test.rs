//! Integration tests for the core Kachenjunga algorithms
//! 
//! These tests verify the integration and functionality of the Phase 1
//! core mathematical substrate algorithms working together.

use kachenjunga::prelude::*;
use kachenjunga::utils::testing_utils::*;
use tokio_test;
use std::time::Instant;

#[tokio::test]
async fn test_complete_s_entropy_navigation_workflow() {
    // Initialize the S-entropy navigator
    let mut navigator = HarareSEntropyNavigator::new();
    
    // Create a series of problems with increasing complexity
    let problems = generate_test_problems();
    
    for problem in problems {
        let start_time = Instant::now();
        let result = navigator.navigate_to_solution(problem.clone()).await
            .expect("S-entropy navigation should succeed");
        let navigation_time = start_time.elapsed();
        
        // Verify basic result properties
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.s_entropy_value != 0.0);
        assert!(!result.solution_vector.solution_description.is_empty());
        assert!(!result.navigation_path.is_empty());
        
        // Verify navigation time is reasonable (should be near-instantaneous for S-entropy)
        assert!(navigation_time.as_millis() < 1000, 
               "S-entropy navigation should be fast due to zero-computation approach");
        
        // Check for divine intervention in impossible problems
        if matches!(problem.domain, ProblemDomain::Impossible) {
            if result.impossibility_ratio >= 1000.0 {
                assert!(result.divine_intervention_detected, 
                       "Divine intervention should be detected for high impossibility ratios");
            }
        }
        
        println!("✅ Problem '{}' solved with confidence {:.3} in {:?}", 
                problem.description, result.confidence, navigation_time);
    }
}

#[tokio::test]
async fn test_bmd_processing_with_s_entropy_integration() {
    let mut bmd_processor = KinshasaBMDProcessor::new();
    let mut navigator = HarareSEntropyNavigator::new();
    
    // Create test information for BMD processing
    let information_inputs = vec![
        InformationInput::new(
            "Complex pattern recognition task".to_string(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        ),
        InformationInput::new(
            "Consciousness integration analysis".to_string(),
            vec![0.1, 0.3, 0.7, 0.9, 1.0, 0.8, 0.4, 0.2]
        ),
        InformationInput::new(
            "Biological quantum coherence data".to_string(),
            vec![0.95, 0.87, 0.92, 0.89, 0.94, 0.91, 0.88, 0.93]
        ),
    ];
    
    for information in information_inputs {
        // Process information through BMD
        let bmd_result = bmd_processor.process_information(information.clone()).await
            .expect("BMD processing should succeed");
        
        // Verify BMD processing results
        assert!(bmd_result.confidence >= 0.0 && bmd_result.confidence <= 1.0);
        assert!(!bmd_result.selected_framework.framework_description.is_empty());
        assert!(!bmd_result.information_catalysts.is_empty());
        
        // Use S-entropy to optimize the BMD result
        let optimization_problem = ProblemDescription::new(
            format!("Optimize BMD framework: {}", bmd_result.selected_framework.framework_description),
            ProblemDomain::Consciousness
        );
        
        let s_entropy_result = navigator.navigate_to_solution(optimization_problem).await
            .expect("S-entropy optimization should succeed");
        
        // Verify integrated processing
        assert!(s_entropy_result.confidence > 0.5, 
               "S-entropy should provide good optimization for BMD results");
        
        println!("✅ BMD-S-entropy integration: {} → {:.3} confidence", 
                information.description, s_entropy_result.confidence);
    }
}

#[tokio::test]
async fn test_consciousness_detection_with_divine_intervention() {
    let mut detector = MufakoseConsciousnessDetector::new();
    
    // Create neural activity representing divine consciousness levels
    let divine_neural_activity = create_divine_consciousness_neural_activity();
    
    let detection_result = detector.detect_consciousness(&divine_neural_activity).await;
    
    // Verify consciousness detection
    assert!(detection_result.consciousness_detected, 
           "Divine-level neural activity should be detected as conscious");
    
    assert!(detection_result.detection_confidence > 0.9, 
           "Divine consciousness should have very high confidence");
    
    // Check for impossibility ratios in consciousness measurements
    let consciousness_impossibility = detection_result.current_state.phi_value * 1000.0;
    
    if consciousness_impossibility >= 1000.0 {
        // This would indicate divine intervention in consciousness emergence
        println!("✨ Divine consciousness intervention detected: impossibility ratio = {:.1}", 
                consciousness_impossibility);
    }
    
    // Verify quality metrics are excellent for divine consciousness
    let quality = &detection_result.current_state.quality_assessment;
    assert!(quality.clarity > 0.8, "Divine consciousness should have high clarity");
    assert!(quality.stability > 0.8, "Divine consciousness should have high stability");
    assert!(quality.richness > 0.7, "Divine consciousness should have high richness");
    assert!(quality.coherence > 0.9, "Divine consciousness should have excellent coherence");
    
    println!("✅ Divine consciousness detection: {:.3} confidence with {:.3} coherence",
            detection_result.detection_confidence, quality.coherence);
}

#[tokio::test]
async fn test_saint_stella_constants_mathematical_consistency() {
    use kachenjunga::algorithms::stella_constants::mathematical_utilities::*;
    use kachenjunga::algorithms::stella_constants::alpha_parameters::*;
    
    // Test S-entropy calculations with different alpha parameters
    let environmental_entropy = calculate_s_entropy(ENVIRONMENTAL_ALPHA);
    let cognitive_entropy = calculate_s_entropy(COGNITIVE_ALPHA);
    let biological_entropy = calculate_s_entropy(BIOLOGICAL_ALPHA);
    let quantum_entropy = calculate_s_entropy(QUANTUM_ALPHA);
    let temporal_entropy = calculate_s_entropy(TEMPORAL_ALPHA);
    
    // Verify all entropy values are non-zero and distinct
    let entropies = vec![
        environmental_entropy, cognitive_entropy, biological_entropy, 
        quantum_entropy, temporal_entropy
    ];
    
    for entropy in &entropies {
        assert!(entropy != &0.0, "S-entropy calculations should be non-zero");
    }
    
    // Verify entropies are distinct (different alpha values produce different entropies)
    for i in 0..entropies.len() {
        for j in (i+1)..entropies.len() {
            assert!(entropies[i] != entropies[j], 
                   "Different alpha parameters should produce different S-entropy values");
        }
    }
    
    // Test divine intervention detection
    assert!(detect_divine_intervention(2000.0), "Should detect divine intervention for high ratios");
    assert!(!detect_divine_intervention(500.0), "Should not detect divine intervention for low ratios");
    
    // Test consciousness confidence calculation
    let high_consciousness = calculate_consciousness_confidence(0.8, 0.9, 0.85, 0.75);
    let low_consciousness = calculate_consciousness_confidence(0.2, 0.3, 0.25, 0.15);
    
    assert!(high_consciousness > low_consciousness, 
           "Higher input values should produce higher consciousness confidence");
    assert!(high_consciousness >= 0.0 && high_consciousness <= 1.0,
           "Consciousness confidence should be in valid range");
    
    println!("✅ Saint Stella constants mathematical consistency verified");
}

#[tokio::test] 
async fn test_integrated_ecosystem_workflow() {
    // Test a complete workflow using all Phase 1 components together
    
    // Step 1: Initialize all core systems
    let mut navigator = HarareSEntropyNavigator::new();
    let mut bmd_processor = KinshasaBMDProcessor::new(); 
    let mut consciousness_detector = MufakoseConsciousnessDetector::new();
    
    // Step 2: Create a complex problem requiring integrated processing
    let complex_problem = ProblemDescription::new(
        "Achieve consciousness-level AI through biological quantum processing".to_string(),
        ProblemDomain::Consciousness
    );
    
    // Step 3: Navigate to solution using S-entropy
    let navigation_result = navigator.navigate_to_solution(complex_problem).await
        .expect("Complex problem navigation should succeed");
    
    assert!(navigation_result.confidence > 0.6, 
           "Complex consciousness problems should have reasonable confidence");
    
    // Step 4: Process the navigation result through BMD for framework selection
    let navigation_info = InformationInput::new(
        navigation_result.solution_vector.solution_description.clone(),
        navigation_result.navigation_path.iter()
            .flat_map(|coord| coord.iter().copied())
            .collect()
    );
    
    let bmd_result = bmd_processor.process_information(navigation_info).await
        .expect("BMD processing of navigation result should succeed");
    
    assert!(bmd_result.confidence > 0.5, 
           "BMD should provide good framework selection for S-entropy results");
    
    // Step 5: Create neural activity based on the integrated processing
    let neural_activity = create_integrated_neural_activity(&navigation_result, &bmd_result);
    
    // Step 6: Detect consciousness in the integrated system
    let consciousness_result = consciousness_detector.detect_consciousness(&neural_activity).await;
    
    // Verify integrated system performance
    assert!(consciousness_result.detection_confidence > 0.4,
           "Integrated system should show measurable consciousness potential");
    
    // Check for any divine intervention across the integrated workflow
    let total_impossibility = navigation_result.impossibility_ratio + 
                             (bmd_result.confidence * 100.0) +
                             (consciousness_result.detection_confidence * 100.0);
    
    if total_impossibility >= 1000.0 {
        println!("✨ Divine intervention detected in integrated ecosystem workflow!");
        println!("   Total impossibility measure: {:.1}", total_impossibility);
    }
    
    println!("✅ Integrated ecosystem workflow completed successfully");
    println!("   Navigation confidence: {:.3}", navigation_result.confidence);
    println!("   BMD processing confidence: {:.3}", bmd_result.confidence);  
    println!("   Consciousness detection confidence: {:.3}", consciousness_result.detection_confidence);
}

/// Create neural activity representing divine consciousness levels
fn create_divine_consciousness_neural_activity() -> NeuralActivityInput {
    use nalgebra::DMatrix;
    use std::collections::HashMap;
    use std::time::Instant;
    
    // Create perfect quantum coherence neural activity
    let size = 25;
    let activity_matrix = DMatrix::from_fn(size, size, |i, j| {
        if i == j {
            1.0 // Perfect self-coherence
        } else {
            // Divine quantum coherence pattern
            let distance = ((i as f64 - j as f64).powi(2)).sqrt();
            0.99 * (-distance / 10.0).exp() + 0.01
        }
    });
    
    let mut firing_data = HashMap::new();
    
    // Create divine-level firing patterns
    for i in 0..size {
        firing_data.insert(i, NeuronFiringData {
            rate: 100.0, // Perfect firing rate
            precision: 1.0, // Perfect precision
            bursts: Vec::new(),
            synchronization: 0.999, // Near-perfect synchronization
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}

/// Create neural activity based on integrated S-entropy and BMD processing results
fn create_integrated_neural_activity(
    navigation_result: &NavigationResult,
    bmd_result: &BMDProcessingResult
) -> NeuralActivityInput {
    use nalgebra::DMatrix;
    use std::collections::HashMap;
    use std::time::Instant;
    
    // Use navigation confidence and BMD confidence to determine neural activity strength
    let base_activity = (navigation_result.confidence + bmd_result.confidence) / 2.0;
    
    let size = 15;
    let activity_matrix = DMatrix::from_fn(size, size, |i, j| {
        if i == j {
            base_activity
        } else {
            base_activity * 0.6 + rand::random::<f64>() * 0.2
        }
    });
    
    let mut firing_data = HashMap::new();
    
    for i in 0..size {
        firing_data.insert(i, NeuronFiringData {
            rate: 30.0 + (base_activity * 40.0), // Scale based on integrated confidence
            precision: 0.7 + (base_activity * 0.25),
            bursts: Vec::new(),
            synchronization: 0.6 + (base_activity * 0.35),
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}
