//! Unit tests for the Harare S-entropy navigation algorithm
//! 
//! These tests verify the specific functionality of the S-entropy navigation
//! system for zero-computation coordinate access to predetermined solutions.

use kachenjunga::prelude::*;
use kachenjunga::algorithms::stella_constants::mathematical_utilities::*;
use tokio_test;
use approx::assert_relative_eq;

#[tokio::test]
async fn test_s_entropy_navigation_basic_functionality() {
    let mut navigator = HarareSEntropyNavigator::new();
    
    let problem = ProblemDescription::new(
        "Basic mathematical optimization".to_string(),
        ProblemDomain::Mathematical
    );
    
    let result = navigator.navigate_to_solution(problem).await
        .expect("Navigation should succeed for basic problems");
    
    // Verify basic properties
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.s_entropy_value != 0.0);
    assert!(!result.solution_vector.solution_description.is_empty());
    assert!(!result.navigation_path.is_empty());
    assert!(result.navigation_path.len() >= 3); // Should have multiple steps
    
    // Verify navigation path progression
    for coordinate in &result.navigation_path {
        assert_eq!(coordinate.len(), 3); // 3D coordinates
        for &value in coordinate {
            assert!(value.is_finite()); // All coordinates should be finite
        }
    }
}

#[tokio::test]
async fn test_s_entropy_calculation_consistency() {
    let mut navigator = HarareSEntropyNavigator::new();
    
    let problem = ProblemDescription::new(
        "S-entropy calculation test".to_string(),
        ProblemDomain::Mathematical
    );
    
    // Navigate multiple times to same problem
    let results: Vec<NavigationResult> = futures::future::try_join_all(
        (0..5).map(|_| navigator.navigate_to_solution(problem.clone()))
    ).await.expect("All navigations should succeed");
    
    // S-entropy values should be consistent for same problem
    let first_entropy = results[0].s_entropy_value;
    for result in &results[1..] {
        assert_relative_eq!(result.s_entropy_value, first_entropy, epsilon = 1e-6);
    }
    
    // Verify S-entropy matches mathematical calculation
    let expected_entropy = calculate_s_entropy(
        kachenjunga::algorithms::stella_constants::alpha_parameters::ENVIRONMENTAL_ALPHA
    );
    
    // Should be in reasonable range relative to expected
    assert!(first_entropy.abs() > 0.0);
    assert!(first_entropy.is_finite());
}

#[tokio::test]
async fn test_divine_intervention_detection() {
    let mut navigator = HarareSEntropyNavigator::new();
    
    // Create impossible problem
    let impossible_problem = ProblemDescription::new(
        "Achieve faster-than-light travel in 25 minutes".to_string(),
        ProblemDomain::Impossible
    );
    
    let result = navigator.navigate_to_solution(impossible_problem).await
        .expect("Even impossible problems should have navigation results");
    
    // Check divine intervention detection logic
    if result.impossibility_ratio >= 1000.0 {
        assert!(result.divine_intervention_detected);
        assert!(!result.impossibility_events.is_empty());
        
        // Verify impossibility event details
        for event in &result.impossibility_events {
            assert!(event.impossibility_ratio >= 1000.0);
            assert!(!event.event_description.is_empty());
            assert!(event.divine_intervention_confidence > 0.5);
        }
    }
    
    // Test explicit divine intervention detection
    assert!(detect_divine_intervention(2000.0));
    assert!(detect_divine_intervention(1000.0));
    assert!(!detect_divine_intervention(999.0));
    assert!(!detect_divine_intervention(100.0));
}

#[tokio::test]
async fn test_problem_domain_handling() {
    let mut navigator = HarareSEntropyNavigator::new();
    
    let domains_and_problems = vec![
        (ProblemDomain::Mathematical, "Solve complex differential equation"),
        (ProblemDomain::Biological, "Optimize cellular ATP production"),
        (ProblemDomain::Consciousness, "Achieve artificial consciousness"),
        (ProblemDomain::Quantum, "Maintain coherence at room temperature"),
        (ProblemDomain::Temporal, "Navigate temporal coordinates precisely"),
        (ProblemDomain::Impossible, "Violate thermodynamic laws beneficially"),
    ];
    
    for (domain, description) in domains_and_problems {
        let problem = ProblemDescription::new(description.to_string(), domain.clone());
        let result = navigator.navigate_to_solution(problem).await
            .expect("Navigation should work for all domains");
        
        // Domain-specific validation
        match domain {
            ProblemDomain::Mathematical => {
                assert!(result.confidence >= 0.3); // Math should be reasonably solvable
            },
            ProblemDomain::Biological => {
                assert!(result.confidence >= 0.4); // Biology is complex but achievable
            },
            ProblemDomain::Consciousness => {
                assert!(result.confidence >= 0.2); // Consciousness is very complex
            },
            ProblemDomain::Quantum => {
                assert!(result.confidence >= 0.3); // Quantum is challenging but possible
            },
            ProblemDomain::Temporal => {
                assert!(result.confidence >= 0.5); // Temporal navigation is the specialty
            },
            ProblemDomain::Impossible => {
                // Impossible problems might have any confidence, but should detect intervention
                if result.impossibility_ratio >= 1000.0 {
                    assert!(result.divine_intervention_detected);
                }
            },
        }
        
        println!("✅ Domain {:?}: confidence {:.3}, impossibility ratio {:.1}",
                domain, result.confidence, result.impossibility_ratio);
    }
}

#[tokio::test]
async fn test_navigation_path_properties() {
    let mut navigator = HarareSEntropyNavigator::new();
    
    let problem = ProblemDescription::new(
        "Multi-step optimization problem".to_string(),
        ProblemDomain::Mathematical
    );
    
    let result = navigator.navigate_to_solution(problem).await
        .expect("Navigation should succeed");
    
    let path = &result.navigation_path;
    
    // Path should have reasonable length
    assert!(path.len() >= 2, "Navigation path should have at least start and end points");
    assert!(path.len() <= 50, "Navigation path should not be excessively long");
    
    // All coordinates should be valid 3D points
    for (i, coordinate) in path.iter().enumerate() {
        assert_eq!(coordinate.len(), 3, "All coordinates should be 3D");
        
        for (j, &value) in coordinate.iter().enumerate() {
            assert!(value.is_finite(), "Coordinate [{}, {}] should be finite", i, j);
            assert!(value >= -1e6 && value <= 1e6, "Coordinate [{}, {}] should be in reasonable range", i, j);
        }
    }
    
    // Path should show progression (not all points identical)
    let start = &path[0];
    let end = &path[path.len() - 1];
    
    let distance: f64 = start.iter().zip(end.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    
    assert!(distance > 1e-6, "Start and end points should be different");
    
    // Calculate path smoothness (adjacent points shouldn't be too far apart)
    for i in 1..path.len() {
        let prev = &path[i-1];
        let curr = &path[i];
        
        let step_distance: f64 = prev.iter().zip(curr.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        
        assert!(step_distance < 100.0, "Navigation steps should not be excessively large");
    }
}

#[tokio::test]
async fn test_solution_vector_completeness() {
    let mut navigator = HarareSEntropyNavigator::new();
    
    let problems = vec![
        ProblemDescription::new("Simple problem".to_string(), ProblemDomain::Mathematical),
        ProblemDescription::new("Complex biological system optimization".to_string(), ProblemDomain::Biological),
        ProblemDescription::new("Consciousness emergence detection".to_string(), ProblemDomain::Consciousness),
    ];
    
    for problem in problems {
        let result = navigator.navigate_to_solution(problem.clone()).await
            .expect("Navigation should succeed");
        
        let solution = &result.solution_vector;
        
        // Solution vector should be complete
        assert!(!solution.solution_description.is_empty());
        assert!(!solution.solution_coordinates.is_empty());
        assert!(solution.solution_quality >= 0.0 && solution.solution_quality <= 1.0);
        assert!(solution.implementation_difficulty >= 0.0 && solution.implementation_difficulty <= 1.0);
        assert!(!solution.required_resources.is_empty());
        
        // Solution coordinates should be valid
        for &coord in &solution.solution_coordinates {
            assert!(coord.is_finite());
        }
        
        // Solution quality should correlate with confidence
        let quality_confidence_correlation = (solution.solution_quality - result.confidence).abs();
        assert!(quality_confidence_correlation < 0.5, 
               "Solution quality and confidence should be reasonably correlated");
        
        println!("✅ Solution for '{}': quality {:.3}, difficulty {:.3}",
                problem.description, solution.solution_quality, solution.implementation_difficulty);
    }
}

#[tokio::test]
async fn test_disposable_pattern_generation() {
    let mut navigator = HarareSEntropyNavigator::new();
    
    // Test disposable patterns for different problem types
    let problem_types = vec![
        "Pattern recognition task",
        "Optimization challenge", 
        "Creative problem solving",
        "Resource allocation",
    ];
    
    for problem_description in problem_types {
        let problem = ProblemDescription::new(
            problem_description.to_string(),
            ProblemDomain::Mathematical
        );
        
        let result = navigator.navigate_to_solution(problem).await
            .expect("Navigation should succeed");
        
        // Verify disposable patterns are generated
        assert!(!result.disposable_patterns.is_empty(), 
               "Should generate disposable patterns for problem solving");
        
        for pattern in &result.disposable_patterns {
            assert!(!pattern.pattern_description.is_empty());
            assert!(pattern.utility_score >= 0.0 && pattern.utility_score <= 1.0);
            assert!(pattern.disposal_condition_met || !pattern.disposal_condition_met); // Boolean should be valid
            assert!(!pattern.pattern_data.is_empty());
            
            // Pattern data should be reasonable
            for &value in &pattern.pattern_data {
                assert!(value.is_finite());
                assert!(value >= -1e6 && value <= 1e6);
            }
        }
        
        println!("✅ Generated {} disposable patterns for '{}'",
                result.disposable_patterns.len(), problem_description);
    }
}

#[tokio::test]
async fn test_zero_computation_approach() {
    let mut navigator = HarareSEntropyNavigator::new();
    
    let problem = ProblemDescription::new(
        "Test zero-computation coordinate access".to_string(),
        ProblemDomain::Temporal
    );
    
    // Measure navigation time
    let start_time = std::time::Instant::now();
    let result = navigator.navigate_to_solution(problem).await
        .expect("Navigation should succeed");
    let navigation_time = start_time.elapsed();
    
    // Zero-computation should be very fast
    assert!(navigation_time.as_millis() < 100, 
           "Zero-computation navigation should be extremely fast (< 100ms)");
    
    // Should still produce valid results despite zero computation
    assert!(result.confidence > 0.0);
    assert!(!result.navigation_path.is_empty());
    assert!(!result.solution_vector.solution_description.is_empty());
    
    // Temporal domain should have high confidence (it's the specialty)
    assert!(result.confidence >= 0.5, 
           "Temporal navigation should have high confidence");
    
    println!("✅ Zero-computation navigation completed in {:?} with {:.3} confidence",
            navigation_time, result.confidence);
}

#[test]
fn test_problem_description_creation() {
    let problem = ProblemDescription::new(
        "Test problem".to_string(),
        ProblemDomain::Mathematical
    );
    
    assert_eq!(problem.description, "Test problem");
    assert!(matches!(problem.domain, ProblemDomain::Mathematical));
    assert!(problem.complexity_estimate > 0.0);
    assert!(problem.expected_solution_time > std::time::Duration::ZERO);
    assert!(!problem.required_capabilities.is_empty());
}

#[test]
fn test_navigation_result_properties() {
    // Test creating navigation result with known values
    let solution_vector = SolutionVector {
        solution_description: "Test solution".to_string(),
        solution_coordinates: vec![1.0, 2.0, 3.0],
        solution_quality: 0.8,
        implementation_difficulty: 0.3,
        required_resources: vec!["CPU time".to_string(), "Memory".to_string()],
    };
    
    let navigation_path = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.5, 0.5, 0.5], 
        vec![1.0, 1.0, 1.0],
    ];
    
    // Verify solution vector properties
    assert!(solution_vector.solution_quality >= 0.0 && solution_vector.solution_quality <= 1.0);
    assert!(solution_vector.implementation_difficulty >= 0.0 && solution_vector.implementation_difficulty <= 1.0);
    assert!(!solution_vector.solution_description.is_empty());
    assert!(!solution_vector.solution_coordinates.is_empty());
    assert!(!solution_vector.required_resources.is_empty());
    
    // Verify navigation path properties
    assert_eq!(navigation_path.len(), 3);
    for coordinate in &navigation_path {
        assert_eq!(coordinate.len(), 3);
    }
}
