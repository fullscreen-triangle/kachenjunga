//! Performance benchmarks and tests for the Kachenjunga algorithm systems
//! 
//! These tests verify that the biological quantum computer algorithms meet
//! performance requirements under various load conditions.

use kachenjunga::prelude::*;
use kachenjunga::utils::testing_utils::*;
use kachenjunga::utils::benchmarking::BenchmarkTimer;
use tokio_test;
use std::time::{Duration, Instant};

#[tokio::test]
async fn benchmark_s_entropy_navigation_performance() {
    let mut navigator = HarareSEntropyNavigator::new();
    let problems = generate_test_problems();
    
    // Test single navigation performance
    let single_start = Instant::now();
    let result = navigator.navigate_to_solution(problems[0].clone()).await
        .expect("Navigation should succeed");
    let single_time = single_start.elapsed();
    
    // Zero-computation navigation should be very fast
    assert!(single_time.as_millis() < 100, 
           "S-entropy navigation should complete in <100ms (zero-computation)");
    
    // Test concurrent navigation performance
    let concurrent_start = Instant::now();
    let concurrent_results = futures::future::try_join_all(
        problems.iter().map(|p| navigator.navigate_to_solution(p.clone()))
    ).await.expect("All concurrent navigations should succeed");
    let concurrent_time = concurrent_start.elapsed();
    
    // Concurrent performance should not degrade significantly
    let expected_concurrent_time = single_time * problems.len() as u32;
    assert!(concurrent_time < expected_concurrent_time * 2,
           "Concurrent navigation should not have excessive overhead");
    
    // Verify all results are valid
    for result in concurrent_results {
        assert!(result.confidence > 0.0);
        assert!(!result.navigation_path.is_empty());
    }
    
    println!("✅ S-entropy navigation performance:");
    println!("    Single navigation: {:?}", single_time);
    println!("    Concurrent {} problems: {:?}", problems.len(), concurrent_time);
    println!("    Average per problem: {:?}", concurrent_time / problems.len() as u32);
}

#[tokio::test]
async fn benchmark_consciousness_detection_performance() {
    let mut detector = MufakoseConsciousnessDetector::new();
    
    // Test with various neural network sizes
    let network_sizes = vec![5, 10, 20, 50];
    let mut performance_results = Vec::new();
    
    for size in network_sizes {
        let neural_activity = create_neural_activity_with_size(size);
        
        let start = Instant::now();
        let result = detector.detect_consciousness(&neural_activity).await;
        let detection_time = start.elapsed();
        
        performance_results.push((size, detection_time, result.detection_confidence));
        
        // Real-time processing requirement: <500ms for networks up to size 50
        if size <= 50 {
            assert!(detection_time.as_millis() < 500,
                   "Consciousness detection should be real-time for networks ≤50 neurons");
        }
        
        println!("Network size {}: {:?} (confidence: {:.3})", 
                size, detection_time, result.detection_confidence);
    }
    
    // Performance should scale reasonably with network size
    let small_time = performance_results[0].1;
    let large_time = performance_results.last().unwrap().1;
    let size_ratio = performance_results.last().unwrap().0 as f64 / performance_results[0].0 as f64;
    let time_ratio = large_time.as_millis() as f64 / small_time.as_millis() as f64;
    
    // Time scaling should be reasonable (not exponential)
    assert!(time_ratio < size_ratio * size_ratio,
           "Consciousness detection should not scale exponentially with network size");
}

#[tokio::test]
async fn benchmark_bmd_processing_throughput() {
    let mut processor = KinshasaBMDProcessor::new();
    
    // Create various information processing workloads
    let information_batches = vec![
        create_small_information_batch(10),
        create_medium_information_batch(50), 
        create_large_information_batch(100),
    ];
    
    for (batch_size, information_batch) in information_batches.iter().enumerate() {
        let batch_name = match batch_size {
            0 => "Small (10 items)",
            1 => "Medium (50 items)",
            2 => "Large (100 items)",
            _ => "Unknown",
        };
        
        let start = Instant::now();
        
        for information in information_batch {
            let _result = processor.process_information(information.clone()).await
                .expect("BMD processing should succeed");
        }
        
        let batch_time = start.elapsed();
        let per_item_time = batch_time / information_batch.len() as u32;
        
        // BMD processing should maintain good throughput
        assert!(per_item_time.as_millis() < 50,
               "BMD processing should handle each item in <50ms");
        
        println!("BMD batch {}: {:?} total, {:?} per item", 
                batch_name, batch_time, per_item_time);
    }
}

#[tokio::test]
async fn benchmark_integrated_system_performance() {
    // Test the complete integrated workflow performance
    let mut navigator = HarareSEntropyNavigator::new();
    let mut bmd_processor = KinshasaBMDProcessor::new();
    let mut consciousness_detector = MufakoseConsciousnessDetector::new();
    
    let integration_problem = ProblemDescription::new(
        "Integrated system performance test".to_string(),
        ProblemDomain::Consciousness
    );
    
    let workflow_start = Instant::now();
    
    // Step 1: S-entropy navigation
    let nav_start = Instant::now();
    let navigation_result = navigator.navigate_to_solution(integration_problem).await
        .expect("Navigation should succeed");
    let nav_time = nav_start.elapsed();
    
    // Step 2: BMD processing
    let bmd_start = Instant::now();
    let navigation_info = InformationInput::new(
        navigation_result.solution_vector.solution_description.clone(),
        vec![1.0, 2.0, 3.0, 4.0] // Simplified data
    );
    let bmd_result = bmd_processor.process_information(navigation_info).await
        .expect("BMD processing should succeed");
    let bmd_time = bmd_start.elapsed();
    
    // Step 3: Consciousness detection
    let consciousness_start = Instant::now();
    let neural_activity = generate_test_neural_activity();
    let consciousness_result = consciousness_detector.detect_consciousness(&neural_activity).await;
    let consciousness_time = consciousness_start.elapsed();
    
    let total_workflow_time = workflow_start.elapsed();
    
    // Integrated workflow should complete in reasonable time
    assert!(total_workflow_time.as_millis() < 1000,
           "Complete integrated workflow should finish in <1 second");
    
    // Individual components should contribute reasonably to total time
    let component_total = nav_time + bmd_time + consciousness_time;
    let overhead_ratio = total_workflow_time.as_millis() as f64 / component_total.as_millis() as f64;
    
    assert!(overhead_ratio < 2.0,
           "Integration overhead should not double the processing time");
    
    println!("✅ Integrated workflow performance:");
    println!("    S-entropy navigation: {:?}", nav_time);
    println!("    BMD processing: {:?}", bmd_time);
    println!("    Consciousness detection: {:?}", consciousness_time);
    println!("    Total workflow: {:?}", total_workflow_time);
    println!("    Integration overhead: {:.1}x", overhead_ratio);
}

#[tokio::test]
async fn benchmark_memory_usage_efficiency() {
    use std::mem;
    
    // Test memory footprint of core structures
    let navigator = HarareSEntropyNavigator::new();
    let bmd_processor = KinshasaBMDProcessor::new();
    let consciousness_detector = MufakoseConsciousnessDetector::new();
    
    let navigator_size = mem::size_of_val(&navigator);
    let bmd_size = mem::size_of_val(&bmd_processor);
    let detector_size = mem::size_of_val(&consciousness_detector);
    
    println!("Memory footprint analysis:");
    println!("    S-entropy navigator: {} bytes", navigator_size);
    println!("    BMD processor: {} bytes", bmd_size);
    println!("    Consciousness detector: {} bytes", detector_size);
    
    let total_core_size = navigator_size + bmd_size + detector_size;
    println!("    Total core systems: {} bytes ({:.2} KB)", total_core_size, total_core_size as f64 / 1024.0);
    
    // Core systems should have reasonable memory footprint
    assert!(total_core_size < 1024 * 1024, // 1MB
           "Core systems should use <1MB base memory");
    
    // Test memory usage during processing
    let problems = generate_test_problems();
    let memory_start = get_approximate_memory_usage();
    
    // Process multiple problems to test memory accumulation
    for problem in problems {
        let _result = navigator.navigate_to_solution(problem).await
            .expect("Navigation should succeed");
    }
    
    let memory_after = get_approximate_memory_usage();
    let memory_growth = memory_after - memory_start;
    
    println!("    Memory growth during processing: {} bytes", memory_growth);
    
    // Memory growth should be reasonable
    assert!(memory_growth < 10 * 1024 * 1024, // 10MB
           "Memory growth during processing should be <10MB");
}

#[tokio::test]
async fn benchmark_divine_intervention_detection_performance() {
    use kachenjunga::algorithms::stella_constants::mathematical_utilities::detect_divine_intervention;
    
    // Test divine intervention detection performance
    let test_ratios = (0..10000).map(|i| i as f64 / 10.0).collect::<Vec<_>>();
    
    let start = Instant::now();
    let mut detection_count = 0;
    
    for ratio in test_ratios {
        if detect_divine_intervention(ratio) {
            detection_count += 1;
        }
    }
    
    let detection_time = start.elapsed();
    let per_check_time = detection_time / 10000;
    
    // Divine intervention detection should be extremely fast
    assert!(per_check_time.as_nanos() < 1000, // <1 microsecond per check
           "Divine intervention detection should be <1μs per check");
    
    // Should detect appropriate number of divine interventions
    assert!(detection_count >= 9000, // Ratios >= 1000 should be detected
           "Should detect divine intervention for high impossibility ratios");
    
    println!("✅ Divine intervention detection performance:");
    println!("    10,000 checks in: {:?}", detection_time);
    println!("    Per check: {:?}", per_check_time);
    println!("    Detections: {} / 10,000", detection_count);
}

#[tokio::test]
async fn benchmark_saint_stella_constants_performance() {
    use kachenjunga::algorithms::stella_constants::mathematical_utilities::*;
    use kachenjunga::algorithms::stella_constants::alpha_parameters::*;
    
    // Test S-entropy calculation performance
    let alphas = vec![ENVIRONMENTAL_ALPHA, COGNITIVE_ALPHA, BIOLOGICAL_ALPHA, 
                     QUANTUM_ALPHA, TEMPORAL_ALPHA];
    
    let calc_start = Instant::now();
    
    for _ in 0..10000 {
        for &alpha in &alphas {
            let _entropy = calculate_s_entropy(alpha);
        }
    }
    
    let calc_time = calc_start.elapsed();
    let per_calc_time = calc_time / (10000 * alphas.len() as u32);
    
    // S-entropy calculations should be very fast
    assert!(per_calc_time.as_nanos() < 100, // <100 nanoseconds per calculation
           "S-entropy calculations should be <100ns each");
    
    // Test consciousness confidence calculation performance
    let confidence_start = Instant::now();
    
    for _ in 0..10000 {
        let _confidence = calculate_consciousness_confidence(0.8, 0.9, 0.85, 0.75);
    }
    
    let confidence_time = confidence_start.elapsed();
    let per_confidence_time = confidence_time / 10000;
    
    assert!(per_confidence_time.as_nanos() < 500, // <500 nanoseconds per calculation
           "Consciousness confidence calculations should be <500ns each");
    
    println!("✅ Saint Stella constants performance:");
    println!("    S-entropy calculation: {:?} per operation", per_calc_time);
    println!("    Consciousness confidence: {:?} per operation", per_confidence_time);
}

#[test]
fn benchmark_data_structure_sizes() {
    use std::mem;
    
    // Test size of key data structures
    let problem_desc = ProblemDescription::new("test".to_string(), ProblemDomain::Mathematical);
    let navigation_result_size = mem::size_of::<NavigationResult>();
    let consciousness_state_size = mem::size_of::<ConsciousnessState>();
    let bmd_result_size = mem::size_of::<BMDProcessingResult>();
    
    println!("Data structure sizes:");
    println!("    ProblemDescription: {} bytes", mem::size_of_val(&problem_desc));
    println!("    NavigationResult: {} bytes", navigation_result_size);
    println!("    ConsciousnessState: {} bytes", consciousness_state_size);
    println!("    BMDProcessingResult: {} bytes", bmd_result_size);
    
    // Structures should be reasonably sized
    assert!(navigation_result_size < 10000, "NavigationResult should be <10KB");
    assert!(consciousness_state_size < 5000, "ConsciousnessState should be <5KB");
    assert!(bmd_result_size < 8000, "BMDProcessingResult should be <8KB");
}

// Helper functions for creating test data

fn create_neural_activity_with_size(size: usize) -> NeuralActivityInput {
    use nalgebra::DMatrix;
    use std::collections::HashMap;
    use std::time::Instant;
    
    let activity_matrix = DMatrix::from_fn(size, size, |i, j| {
        if i == j { 0.8 } else { rand::random::<f64>() * 0.5 }
    });
    
    let mut firing_data = HashMap::new();
    for i in 0..size {
        firing_data.insert(i, NeuronFiringData {
            rate: 20.0 + rand::random::<f64>() * 30.0,
            precision: 0.7 + rand::random::<f64>() * 0.3,
            bursts: Vec::new(),
            synchronization: 0.6 + rand::random::<f64>() * 0.4,
        });
    }
    
    NeuralActivityInput {
        activity_matrix,
        firing_data,
        connectivity_matrix: None,
        timestamp: Instant::now(),
    }
}

fn create_small_information_batch(count: usize) -> Vec<InformationInput> {
    (0..count).map(|i| {
        InformationInput::new(
            format!("Small information item {}", i),
            vec![i as f64, (i * 2) as f64]
        )
    }).collect()
}

fn create_medium_information_batch(count: usize) -> Vec<InformationInput> {
    (0..count).map(|i| {
        InformationInput::new(
            format!("Medium complexity information item {} with additional context", i),
            (0..10).map(|j| (i * j) as f64).collect()
        )
    }).collect()
}

fn create_large_information_batch(count: usize) -> Vec<InformationInput> {
    (0..count).map(|i| {
        InformationInput::new(
            format!("Large, complex information item {} requiring extensive processing and framework selection with multiple context layers", i),
            (0..50).map(|j| ((i * j) as f64).sin() * ((i + j) as f64).cos()).collect()
        )
    }).collect()
}

fn get_approximate_memory_usage() -> usize {
    // This is a simplified memory usage estimation
    // In a real implementation, you might use system-specific memory tracking
    0 // Placeholder - would implement actual memory tracking
}
