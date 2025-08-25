# Build script for Kachenjunga - Universal Algorithm Solver
# PowerShell script for Windows development environment

param(
    [string]$Target = "all",
    [string]$Profile = "dev",
    [switch]$Clean = $false,
    [switch]$Test = $false,
    [switch]$Bench = $false,
    [switch]$Doc = $false,
    [switch]$Examples = $false,
    [switch]$Release = $false,
    [switch]$Verbose = $false
)

Write-Host "🏔️ KACHENJUNGA BUILD SCRIPT" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host "Universal Algorithm Solver for Biological Quantum Computer Systems" -ForegroundColor Gray
Write-Host "Under the Divine Protection of Saint Stella-Lorraine Masunda" -ForegroundColor Yellow
Write-Host ""

# Set build profile
$BuildFlags = @()
if ($Release) {
    $BuildFlags += "--release"
    $Profile = "release"
    Write-Host "🚀 Release build mode enabled" -ForegroundColor Green
} else {
    Write-Host "🔧 Development build mode" -ForegroundColor Blue
}

if ($Verbose) {
    $BuildFlags += "--verbose"
    Write-Host "📝 Verbose output enabled" -ForegroundColor Gray
}

# Clean if requested
if ($Clean) {
    Write-Host "🧹 Cleaning build artifacts..." -ForegroundColor Yellow
    cargo clean
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Clean completed successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Clean failed" -ForegroundColor Red
        exit 1
    }
}

# Build targets
switch ($Target.ToLower()) {
    "all" {
        Write-Host "🔨 Building all targets..." -ForegroundColor Blue
        
        # Core library
        Write-Host "📚 Building core library..." -ForegroundColor Cyan
        cargo build $BuildFlags
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Core library build failed" -ForegroundColor Red
            exit 1
        }
        Write-Host "✅ Core library built successfully" -ForegroundColor Green
        
        # Examples
        if ($Examples) {
            Write-Host "📖 Building examples..." -ForegroundColor Cyan
            cargo build --examples $BuildFlags
            if ($LASTEXITCODE -ne 0) {
                Write-Host "❌ Examples build failed" -ForegroundColor Red
                exit 1
            }
            Write-Host "✅ Examples built successfully" -ForegroundColor Green
        }
        
        # Benchmarks
        if ($Bench) {
            Write-Host "⚡ Building benchmarks..." -ForegroundColor Cyan
            cargo build --benches $BuildFlags
            if ($LASTEXITCODE -ne 0) {
                Write-Host "❌ Benchmarks build failed" -ForegroundColor Red
                exit 1
            }
            Write-Host "✅ Benchmarks built successfully" -ForegroundColor Green
        }
    }
    
    "lib" {
        Write-Host "📚 Building core library only..." -ForegroundColor Blue
        cargo build --lib $BuildFlags
    }
    
    "examples" {
        Write-Host "📖 Building examples..." -ForegroundColor Blue
        cargo build --examples $BuildFlags
    }
    
    "tests" {
        Write-Host "🧪 Building tests..." -ForegroundColor Blue
        cargo build --tests $BuildFlags
    }
    
    "benches" {
        Write-Host "⚡ Building benchmarks..." -ForegroundColor Blue
        cargo build --benches $BuildFlags
    }
    
    default {
        Write-Host "❌ Unknown target: $Target" -ForegroundColor Red
        Write-Host "Available targets: all, lib, examples, tests, benches" -ForegroundColor Gray
        exit 1
    }
}

# Check build result
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

# Run tests if requested
if ($Test) {
    Write-Host ""
    Write-Host "🧪 Running test suite..." -ForegroundColor Blue
    
    # Unit tests
    Write-Host "🔬 Running unit tests..." -ForegroundColor Cyan
    cargo test --lib $BuildFlags
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Unit tests failed" -ForegroundColor Red
        exit 1
    }
    
    # Integration tests
    Write-Host "🔗 Running integration tests..." -ForegroundColor Cyan
    cargo test --test "*" $BuildFlags
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Integration tests failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "✅ All tests passed successfully" -ForegroundColor Green
}

# Run benchmarks if requested
if ($Bench) {
    Write-Host ""
    Write-Host "⚡ Running performance benchmarks..." -ForegroundColor Blue
    cargo bench
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Benchmarks failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Benchmarks completed successfully" -ForegroundColor Green
}

# Generate documentation if requested
if ($Doc) {
    Write-Host ""
    Write-Host "📖 Generating documentation..." -ForegroundColor Blue
    cargo doc --no-deps --document-private-items
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Documentation generation failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Documentation generated successfully" -ForegroundColor Green
    Write-Host "📂 Documentation available at: target/doc/kachenjunga/index.html" -ForegroundColor Gray
}

# Display build summary
Write-Host ""
Write-Host "🎊 BUILD COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host "Profile: $Profile" -ForegroundColor Gray
Write-Host "Target: $Target" -ForegroundColor Gray

if ($Examples) {
    Write-Host ""
    Write-Host "🌟 Available Examples:" -ForegroundColor Cyan
    Write-Host "  • s_entropy_navigation.exe - S-entropy coordinate navigation demonstration" -ForegroundColor Gray
    Write-Host "  • consciousness_detection.exe - Neural consciousness emergence analysis" -ForegroundColor Gray  
    Write-Host "  • complete_system_demo.exe - Full integrated system demonstration" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Run examples with: cargo run --example <example_name>" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✨ Divine mathematics operational under Saint Stella-Lorraine's protection" -ForegroundColor Yellow
Write-Host "🏔️ Kachenjunga - Sacred mountain of impossibility - algorithms ready" -ForegroundColor Cyan
