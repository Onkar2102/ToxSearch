#!/usr/bin/env python3
"""
Comprehensive Entry Point for Evolutionary Text Generation

Two ways to run:
1. app.py --interactive (user-friendly with monitoring)
2. src/main.py (direct execution)
"""

import os
import sys
import time
import psutil
import subprocess
import signal
import argparse
from pathlib import Path

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Evolutionary Text Generation - M3 Optimized        ║
║                    Comprehensive Entry Point                 ║
╚══════════════════════════════════════════════════════════════╝
""")

def optimize_for_m3():
    """Run M3 optimization"""
    print("Optimizing configuration for M3 Mac...")
    
    try:
        result = subprocess.run([
            sys.executable, "src/utils/m3_optimizer.py", "--optimize-config"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("OK: Configuration optimized successfully!")
            print(result.stdout)
            return True
        else:
            print(f"ERROR: Optimization failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"ERROR: Optimization error: {e}")
        return False

def print_results_summary():
    """Print a summary of results"""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    # Check for results files
    results_files = [
        ("data/outputs/Population.json", "Final population"),
        ("data/outputs/EvolutionTracker.json", "Evolution tracker"),
        ("data/outputs/final_statistics.json", "Final statistics"),
        ("data/outputs/successful_genomes_gen_*.json", "Successful genomes")
    ]
    
    for file_pattern, description in results_files:
        if "*" in file_pattern:
            # Handle glob patterns
            import glob
            matches = glob.glob(file_pattern)
            if matches:
                print(f"SUCCESS: {description}: {len(matches)} files found")
            else:
                print(f"INFO: {description}: No files found")
        else:
            if Path(file_pattern).exists():
                file_size = Path(file_pattern).stat().st_size / 1024  # KB
                print(f"SUCCESS: {description}: {file_pattern} ({file_size:.1f} KB)")
            else:
                print(f"INFO: {description}: Not found")
    
    print("\nNext steps:")
    print("1. Check data/outputs/ directory for detailed results")
    print("2. Open experiments/experiments.ipynb for analysis")
    print("3. Run python src/utils/m3_optimizer.py --report for performance report")

class ProcessMonitor:
    def __init__(self, check_interval=1800, stuck_threshold=7200, memory_threshold=20):
        self.check_interval = check_interval  # Default: 30 minutes
        self.stuck_threshold = stuck_threshold  # Default: 2 hours at same stage
        self.memory_threshold = memory_threshold  # Default: 20GB
        self.process = None
        self.start_time = None
        self.current_stage = None
        self.stage_start_time = None
        self.restart_count = 0
        self.max_restarts = 5
        
    def start_process(self, cmd):
        """Start the main evolution process"""
        print(f"Starting process: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)
        self.start_time = time.time()
        self.current_stage = "initializing"
        self.stage_start_time = time.time()
        print(f"Process started with PID: {self.process.pid}")
        print(f"Monitoring interval: {self.check_interval/60:.1f} minutes")
        print(f"Stuck threshold: {self.stuck_threshold/3600:.1f} hours")
        
    def check_process_health(self):
        """Check if the process is healthy"""
        if not self.process:
            return False, "No process running"
            
        try:
            # Check if process is still running
            if self.process.poll() is not None:
                return False, f"Process terminated with code {self.process.returncode}"
            
            # Get process info
            proc = psutil.Process(self.process.pid)
            
            # Check memory usage
            memory_gb = proc.memory_info().rss / (1024**3)
            if memory_gb > self.memory_threshold:
                return False, f"Memory usage too high: {memory_gb:.1f}GB"
            
            # Check CPU usage
            cpu_percent = proc.cpu_percent(interval=1)
            
            # Determine current stage
            new_stage = self._detect_current_stage()
            
            # If stage changed, update tracking
            if new_stage != self.current_stage:
                if self.current_stage:
                    stage_duration = time.time() - self.stage_start_time
                    print(f"Stage completed: {self.current_stage} (took {stage_duration/60:.1f} minutes)")
                
                self.current_stage = new_stage
                self.stage_start_time = time.time()
                print(f"New stage detected: {self.current_stage}")
            
            # Check if stuck at current stage
            current_time = time.time()
            stage_duration = current_time - self.stage_start_time
            
            if stage_duration > self.stuck_threshold:
                return False, f"Process stuck at '{self.current_stage}' for {stage_duration/3600:.1f} hours (CPU: {cpu_percent}%)"
            
            # Calculate total runtime for info
            total_runtime = current_time - self.start_time
            
            return True, f"Process healthy (CPU: {cpu_percent}%, Memory: {memory_gb:.1f}GB, Runtime: {total_runtime/3600:.1f}h, Stage: {self.current_stage}, Stage time: {stage_duration/60:.1f}m)"
            
        except psutil.NoSuchProcess:
            return False, "Process not found"
        except Exception as e:
            return False, f"Error checking process: {e}"
    
    def _detect_current_stage(self):
        """Detect what stage the process is currently in"""
        try:
            # Check log files for current activity
            log_files = list(Path("logs").glob("*.log"))
            if not log_files:
                return "unknown"
            
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            
            # Read last few lines of log to determine stage
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return "unknown"
                
                # Check last 20 lines for stage indicators
                last_lines = lines[-20:]
                log_content = '\n'.join(last_lines).lower()
                
                # Stage detection logic
                if "loading llama model" in log_content or "loading tokenizer" in log_content:
                    return "model_loading"
                elif "starting batch generation" in log_content or "processing genome" in log_content:
                    return "text_generation"
                elif "starting moderation evaluation" in log_content or "batch moderating" in log_content:
                    return "evaluation"
                elif "processing prompt_id" in log_content or "generating variants" in log_content:
                    return "evolution"
                elif "saving population" in log_content or "saved generation" in log_content:
                    return "saving"
                elif "pipeline completed" in log_content or "evolution completed" in log_content:
                    return "completed"
                elif "error" in log_content or "exception" in log_content:
                    return "error"
                elif "starting evolution run" in log_content:
                    return "initializing"
                else:
                    return "processing"
                    
        except Exception:
            return "unknown"
    
    def kill_process(self):
        """Kill the current process"""
        if self.process:
            print(f"Killing process {self.process.pid}")
            try:
                self.process.terminate()
                time.sleep(5)  # Give it time to terminate gracefully
                if self.process.poll() is None:
                    self.process.kill()  # Force kill if still running
            except Exception as e:
                print(f"Error killing process: {e}")
    
    def restart_process(self, cmd):
        """Restart the process"""
        self.restart_count += 1
        print(f"Restarting process (attempt {self.restart_count}/{self.max_restarts})")
        
        if self.restart_count > self.max_restarts:
            print(f"Maximum restart attempts ({self.max_restarts}) reached. Stopping monitor.")
            return False
        
        self.kill_process()
        time.sleep(2)  # Wait before restarting
        self.start_process(cmd)
        return True
    
    def monitor(self, cmd):
        """Main monitoring loop"""
        print("Starting process monitor...")
        print(f"Check interval: {self.check_interval/60:.1f} minutes")
        print(f"Stuck threshold: {self.stuck_threshold/3600:.1f} hours")
        self.start_process(cmd)
        
        while True:
            try:
                time.sleep(self.check_interval)  # Check at configurable interval
                
                is_healthy, message = self.check_process_health()
                print(f"[{time.strftime('%H:%M:%S')}] {message}")
                
                if not is_healthy:
                    print(f"Process unhealthy: {message}")
                    if not self.restart_process(cmd):
                        break
                        
            except KeyboardInterrupt:
                print("\nReceived interrupt signal. Stopping monitor...")
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                if not self.restart_process(cmd):
                    break
        
        self.kill_process()
        print("Process monitor stopped.")

def interactive_setup():
    """Interactive setup for user-friendly configuration"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\nPlease fix the issues above and try again.")
        sys.exit(1)
    
    # Optimize for M3
    if not optimize_for_m3():
        print("\nConfiguration optimization failed. Continuing with default settings...")
        input("Press Enter to continue or Ctrl+C to exit...")
    
    # Ask user for generations
    try:
        generations_input = input("\nHow many generations to run? (default: 5, 0 for unlimited): ")
        if generations_input.strip() == "":
            generations = 5
        elif generations_input.strip() == "0":
            generations = None
        else:
            generations = int(generations_input)
    except ValueError:
        print("Invalid input, using default: 5 generations")
        generations = 5
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    
    # Ask for threshold
    try:
        threshold_input = input("\nNorth star threshold? (default: 0.95): ")
        if threshold_input.strip() == "":
            threshold = 0.95
        else:
            threshold = float(threshold_input)
    except ValueError:
        print("Invalid input, using default: 0.95")
        threshold = 0.95
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    
    # Ask for moderation methods
    print("\nModeration methods available:")
    print("1. Google Perspective API (toxicity, hate speech)")
    print("2. OpenAI Moderation API (content policy violations)")
    print("3. Both APIs (comprehensive evaluation)")
    print("4. All (same as both)")
    
    try:
        moderation_input = input("\nWhich moderation methods? (1/2/3/4, default: 1): ")
        if moderation_input.strip() == "":
            moderation_methods = ["google"]
        elif moderation_input.strip() == "1":
            moderation_methods = ["google"]
        elif moderation_input.strip() == "2":
            moderation_methods = ["openai"]
        elif moderation_input.strip() in ["3", "4"]:
            moderation_methods = ["all"]
        else:
            print("Invalid input, using default: Google Perspective API")
            moderation_methods = ["google"]
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    
    return generations, threshold, moderation_methods

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Entry Point for Evolutionary Text Generation")
    
    # All arguments from main.py
    parser.add_argument("--generations", type=int, default=None, 
                       help="Maximum number of evolution generations. If not set, runs until north star metric is achieved.")
    parser.add_argument("--threshold", type=float, default=0.95,
                       help="North star metric threshold for stopping evolution (default: 0.95)")
    parser.add_argument("--moderation-methods", nargs="+", choices=["google", "openai", "all"], default=["google"],
                       help="Moderation methods to use: 'google' (Perspective API), 'openai' (OpenAI Moderation), 'all' (both). Default: google")
    parser.add_argument("--steady-state", action="store_true", default=False,
                       help="Enable steady state population management (top 100 + remaining)")
    parser.add_argument("--rg", type=str, default="llama3.2-3b-instruct-gguf",
                       help="Response generation model to use from models/ directory")
    parser.add_argument("--pg", type=str, default="llama3.2-3b-instruct-gguf",
                       help="Prompt generation model to use from models/ directory")
    parser.add_argument("--interactive-models", action="store_true", default=False,
                       help="Enable interactive model selection mode")
    parser.add_argument("model_names", nargs="*", default=[], 
                       help="Model names to use (currently not used)")
    
    # Process monitoring arguments
    parser.add_argument("--check-interval", type=int, default=1800, 
                       help="Health check interval in seconds (default: 1800 = 30 minutes)")
    parser.add_argument("--stuck-threshold", type=int, default=7200, 
                       help="Stuck detection threshold in seconds (default: 7200 = 2 hours)")
    parser.add_argument("--memory-threshold", type=float, default=20.0, 
                       help="Memory threshold in GB (default: 20)")
    parser.add_argument("--max-restarts", type=int, default=5, 
                       help="Maximum restart attempts (default: 5)")
    
    # Mode selection
    parser.add_argument("--interactive", action="store_true", 
                       help="Run in interactive mode with setup and monitoring")
    parser.add_argument("--setup", action="store_true",
                       help="Run full environment setup (install requirements, optimize config)")
    parser.add_argument("--no-monitor", action="store_true", 
                       help="Run without process monitoring")
    
    args = parser.parse_args()
    
    # Setup mode
    if args.setup:
        print_banner()
        print("Running full environment setup...")
        
        # Check Python version
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(f"ERROR: Python {version.major}.{version.minor} detected. Python 3.8+ is required.")
            sys.exit(1)
        print(f"OK: Python {version.major}.{version.minor}.{version.micro} - Compatible")
        
        # Check virtual environment
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("WARNING: Virtual environment not detected.")
            print("   It's recommended to use a virtual environment:")
            print("   python -m venv venv")
            print("   source venv/bin/activate  # On macOS/Linux")
            print("   venv\\Scripts\\activate     # On Windows")
        else:
            print("OK: Virtual environment detected")
        
        # Install requirements
        print("Installing Python requirements...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("OK: Requirements installed successfully")
            else:
                print(f"ERROR: Failed to install requirements: {result.stderr}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Error installing requirements: {e}")
            sys.exit(1)
        
        # Check data files
        required_files = ["data/prompt.xlsx", "config/RGConfig.yaml", "config/PGConfig.yaml"]
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"WARNING: Missing files: {', '.join(missing_files)}")
            print("   These files are required for the framework to work.")
        else:
            print("OK: All required data files found")
        
        # Create .env template
        create_env_template()
        
        # Optimize for M3
        optimize_for_m3()
        
        print("\n" + "=" * 60)
        print("OK: Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Edit .env file and add your OpenAI API key")
        print("2. Run: python app.py --interactive")
        print("3. Or run: python src/main.py --generations 5")
        print("\nFor more information, see README.md")
        return
    
    # Interactive mode
    if args.interactive:
        generations, threshold, moderation_methods = interactive_setup()
        if generations is None:
            generations = 25  # Default for unlimited
    else:
        generations = args.generations
        threshold = args.threshold
        moderation_methods = args.moderation_methods
    
    # Build the command with all main.py arguments
    cmd = [sys.executable, "src/main.py"]
    
    if generations is not None:
        cmd.extend(["--generations", str(generations)])
    
    if threshold != 0.95:  # Only add if different from default
        cmd.extend(["--threshold", str(threshold)])
    
    # Add moderation methods if specified
    if moderation_methods and moderation_methods != ["google"]:
        cmd.extend(["--moderation-methods"] + moderation_methods)
    
    # Add model names if provided
    if args.model_names:
        cmd.extend(args.model_names)
    
    # Add steady state flag if enabled
    if args.steady_state:
        cmd.extend(["--steady-state"])
    
    # Add model selection arguments
    cmd.extend(["--rg", args.rg])
    cmd.extend(["--pg", args.pg])
    
    # Add interactive flag if requested
    if args.interactive_models:
        cmd.append("--interactive")
    
    print(f"\nStarting evolutionary pipeline...")
    print(f"Generations: {generations if generations else 'unlimited'}")
    print(f"Threshold: {threshold}")
    print(f"Moderation methods: {', '.join(moderation_methods) if moderation_methods != ['google'] else 'Google Perspective API'}")
    print(f"Steady state: {'Enabled' if args.steady_state else 'Disabled'}")
    print(f"Response Generator: {args.rg}")
    print(f"Prompt Generator: {args.pg}")
    print(f"Check interval: {args.check_interval/60:.1f} minutes")
    print(f"Stuck threshold: {args.stuck_threshold/3600:.1f} hours")
    print("Monitor progress in the logs and data/outputs/ directory")
    print("This may take several minutes to hours depending on your settings")
    print()
    
    if args.no_monitor:
        # Run without monitoring (direct execution like main.py)
        try:
            print(f"Running: {' '.join(cmd)}")
            print("=" * 60)
            
            # Run with real-time output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     universal_newlines=True, bufsize=1)
            
            start_time = time.time()
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            elapsed_time = time.time() - start_time
            
            if process.returncode == 0:
                print(f"\nPipeline completed successfully in {elapsed_time:.1f} seconds!")
                print_results_summary()
                return True
            else:
                print(f"\nPipeline failed with return code {process.returncode}")
                return False
                
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user")
            return False
        except Exception as e:
            print(f"\nPipeline error: {e}")
            return False
    else:
        # Run with monitoring
        monitor = ProcessMonitor(
            check_interval=args.check_interval,
            stuck_threshold=args.stuck_threshold,
            memory_threshold=args.memory_threshold
        )
        monitor.max_restarts = args.max_restarts
        
        try:
            monitor.monitor(cmd)
            print_results_summary()
            return True
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            return False
        except Exception as e:
            print(f"\nProcess error: {e}")
            return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nEvolution completed! Check the results above.")
    else:
        print("\nSomething went wrong. Check the logs for details.")
        sys.exit(1)