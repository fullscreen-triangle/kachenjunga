use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kachenjunga::prelude::*;
use tokio::runtime::Runtime;

fn benchmark_s_entropy_navigation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut navigator = HarareSEntropyNavigator::new();
    
    let problem = ProblemDescription::new(
        "Benchmark test problem".to_string(),
        ProblemDomain::Mathematical
    );
    
    c.bench_function("s_entropy_navigation", |b| {
        b.to_async(&rt).iter(|| async {
            let problem_clone = black_box(problem.clone());
            let result = navigator.navigate_to_solution(problem_clone).await;
            black_box(result)
        })
    });
}

fn benchmark_consciousness_detection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut detector = MufakoseConsciousnessDetector::new();
    let neural_activity = kachenjunga::utils::testing_utils::generate_test_neural_activity();
    
    c.bench_function("consciousness_detection", |b| {
        b.to_async(&rt).iter(|| async {
            let activity_clone = black_box(neural_activity.clone());
            let result = detector.detect_consciousness(&activity_clone).await;
            black_box(result)
        })
    });
}

fn benchmark_bmd_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut processor = KinshasaBMDProcessor::new();
    let information = InformationInput::new(
        "Test information".to_string(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0]
    );
    
    c.bench_function("bmd_processing", |b| {
        b.to_async(&rt).iter(|| async {
            let info_clone = black_box(information.clone());
            let result = processor.process_information(info_clone).await;
            black_box(result)
        })
    });
}

criterion_group!(benches, 
    benchmark_s_entropy_navigation,
    benchmark_consciousness_detection, 
    benchmark_bmd_processing
);
criterion_main!(benches);
