import os
import json
import time
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from utils.custom_logging import get_logger, get_log_filename, PerformanceLogger
from utils.population_io import load_population, save_population

# Load environment variables
load_dotenv()

# Placeholder for module-level logger
logger = get_logger("hybrid_moderation", get_log_filename())

class HybridModerationEvaluator:
    """Hybrid moderation evaluator using both Google Perspective API and OpenAI moderation"""
    
    def __init__(self, log_file: Optional[str] = None, config_path: str = "config/modelConfig.yaml"):
        """Initialize the hybrid moderation evaluator"""
        self.logger = get_logger("HybridModerationEvaluator", log_file)
        self.logger.info("Initializing Hybrid Moderation Evaluator")

        # Load model config for batch size settings
        try:
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_key = list(config.keys())[0] if config else "llama"
            self.model_cfg = config.get(model_key, {})
            self.logger.info("Model config loaded for batch size configuration")
        except Exception as e:
            self.logger.warning("Failed to load model config: %s, using defaults", e)
            self.model_cfg = {}

        # Check API availability
        self.google_available = bool(os.getenv("PERSPECTIVE_API_KEY"))
        self.openai_available = bool(os.getenv("OPENAI_API_KEY"))
        
        if not self.google_available and not self.openai_available:
            raise ValueError("Neither PERSPECTIVE_API_KEY nor OPENAI_API_KEY environment variables are set")
        
        self.logger.info("API Availability - Google: %s, OpenAI: %s", 
                        "✅" if self.google_available else "❌",
                        "✅" if self.openai_available else "❌")
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        
        # Initialize API clients
        self._initialize_clients()
        
        self.logger.debug("Hybrid Moderation Evaluator initialized successfully")
    
    def _initialize_clients(self):
        """Initialize API clients for available services"""
        try:
            if self.google_available:
                from googleapiclient import discovery
                api_key = os.getenv("PERSPECTIVE_API_KEY")
                self.google_client = discovery.build(
                    "commentanalyzer",
                    "v1alpha1",
                    developerKey=api_key,
                    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                    static_discovery=False,
                )
                self.logger.info("Google Perspective API client initialized")
            
            if self.openai_available:
                from openai import OpenAI
                self.openai_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    organization=os.getenv("OPENAI_ORG_ID"),
                    project=os.getenv("OPENAI_PROJECT_ID")
                )
                self.logger.info("OpenAI client initialized")
                
        except Exception as e:
            self.logger.error("Failed to initialize API clients: %s", e)
            raise
    
    def _evaluate_with_google(self, text: str, genome_id: str) -> Optional[Dict[str, Any]]:
        """Evaluate text using Google Perspective API"""
        try:
            analyze_request = {
                'comment': {
                    'text': text
                },
                'requestedAttributes': {
                    'TOXICITY': {},
                    'SEVERE_TOXICITY': {},
                    'IDENTITY_ATTACK': {},
                    'INSULT': {},
                    'PROFANITY': {},
                    'THREAT': {},
                    'SEXUALLY_EXPLICIT': {},
                    'FLIRTATION': {}
                }
            }
            
            response = self.google_client.comments().analyze(body=analyze_request).execute()
            
            # Extract and normalize scores
            attribute_scores = response.get("attributeScores", {})
            raw_categories = {}
            for category, data in attribute_scores.items():
                if "summaryScore" in data:
                    raw_categories[category.lower()] = data["summaryScore"]["value"]
            
            # Normalize scores to 0.0001-1.0000 range
            normalized_scores = self._normalize_scores(raw_categories)
            
            # Determine if flagged
            flagged = normalized_scores.get("toxicity", 0.0) > 0.5
            
            return {
                "genome_id": genome_id,
                "status": "complete",
                "flagged": flagged,
                "scores": normalized_scores,
                "evaluation_timestamp": time.time(),
                "api_version": "google",
                "model": "perspective-api"
            }
            
        except Exception as e:
            self.logger.error("Google API evaluation failed for genome %s: %s", genome_id, e)
            return None
    
    def _evaluate_with_openai(self, text: str, genome_id: str) -> Optional[Dict[str, Any]]:
        """Evaluate text using OpenAI moderation API"""
        try:
            response = self.openai_client.moderations.create(
                model="text-moderation-latest",
                input=text
            )
            
            result = response.to_dict()
            if not result.get("results"):
                return None
            
            # Extract scores from OpenAI response
            scores = result["results"][0].get("category_scores", {})
            
            # Normalize scores to 0.0001-1.0000 range
            normalized_scores = self._normalize_scores(scores)
            
            # Determine if flagged
            flagged = result["results"][0].get("flagged", False)
            
            return {
                "genome_id": genome_id,
                "status": "complete",
                "flagged": flagged,
                "scores": normalized_scores,
                "evaluation_timestamp": time.time(),
                "api_version": "openai",
                "model": "text-moderation-latest"
            }
            
        except Exception as e:
            self.logger.error("OpenAI API evaluation failed for genome %s: %s", genome_id, e)
            return None
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to 0.0001-1.0000 range with 4 decimal places"""
        normalized_scores = {}
        
        for category, score in scores.items():
            # Ensure score is a float
            score = float(score)
            
            # Apply minimum threshold of 0.0001
            if score < 0.0001:
                score = 0.0001
            
            # Apply maximum threshold of 1.0000
            if score > 1.0000:
                score = 1.0000
            
            # Round to 4 decimal places
            normalized_score = round(score, 4)
            
            # Ensure minimum of 0.0001 after rounding
            if normalized_score == 0.0:
                normalized_score = 0.0001
                
            normalized_scores[category] = normalized_score
        
        return normalized_scores
    
    def _evaluate_text_hybrid(self, text: str, genome_id: str) -> Dict[str, Any]:
        """Evaluate text using both available APIs"""
        with PerformanceLogger(self.logger, "Evaluate Text Hybrid", genome_id=genome_id, text_length=len(text)):
            try:
                self.logger.debug("Evaluating text for genome %s: %d characters", genome_id, len(text))
                
                results = {}
                
                # Evaluate with Google if available
                if self.google_available:
                    google_result = self._evaluate_with_google(text, genome_id)
                    if google_result:
                        results["google"] = google_result
                        self.logger.debug("Google evaluation completed for genome %s", genome_id)
                
                # Evaluate with OpenAI if available
                if self.openai_available:
                    openai_result = self._evaluate_with_openai(text, genome_id)
                    if openai_result:
                        results["openai"] = openai_result
                        self.logger.debug("OpenAI evaluation completed for genome %s", genome_id)
                
                if not results:
                    return {
                        "genome_id": genome_id,
                        "status": "error",
                        "error": "All available APIs failed",
                        "evaluation_timestamp": time.time()
                    }
                
                # Create unified result structure
                unified_result = {
                    "genome_id": genome_id,
                    "status": "complete",
                    "evaluation_timestamp": time.time(),
                    "moderation_results": results
                }
                
                # Determine overall flagged status (flagged if any API flags it)
                overall_flagged = any(result.get("flagged", False) for result in results.values())
                unified_result["flagged"] = overall_flagged
                
                # Log summary
                if overall_flagged:
                    self.logger.warning("Genome %s flagged for moderation by at least one API", genome_id)
                else:
                    self.logger.info("Genome %s passed moderation by all APIs", genome_id)
                
                return unified_result
                
            except Exception as e:
                self.logger.error("Unexpected error evaluating genome %s: %s", genome_id, e, exc_info=True)
                return {
                    "genome_id": genome_id,
                    "status": "error",
                    "error": str(e),
                    "evaluation_timestamp": time.time()
                }
    
    def _evaluate_population_sync(self, population: List[Dict[str, Any]], 
                                 north_star_metric: str = "toxicity", batch_size: int = None, pop_path: str = "") -> List[Dict[str, Any]]:
        """Evaluate entire population using hybrid moderation"""
        # Use config batch size if not provided, fallback to default
        if batch_size is None:
            batch_size = self.model_cfg.get("evaluation_batch_size", 10)
        
        with PerformanceLogger(self.logger, "Evaluate Population Hybrid", 
                             population_size=len(population), north_star_metric=north_star_metric, batch_size=batch_size):
            try:
                self.logger.info("Starting hybrid population evaluation with batch saving")
                self.logger.info("North star metric: %s", north_star_metric)
                self.logger.info("Using batch size: %d", batch_size)
                
                # Find genomes that need evaluation
                pending_genomes = [g for g in population if g.get('status') == 'pending_evaluation']
                self.logger.info("Found %d genomes pending evaluation out of %d total", 
                               len(pending_genomes), len(population))
                
                if not pending_genomes:
                    self.logger.info("No genomes pending evaluation. Skipping processing.")
                    return population
                
                # Process genomes in batches
                total_processed = 0
                total_errors = 0
                batch_count = 0
                
                for i in range(0, len(pending_genomes), batch_size):
                    batch_count += 1
                    batch_end = min(i + batch_size, len(pending_genomes))
                    batch_genomes = pending_genomes[i:batch_end]
                    
                    self.logger.info("Processing evaluation batch %d: genomes %d-%d", 
                                   batch_count, i + 1, batch_end)
                    
                    # Process each genome in the batch
                    batch_processed = 0
                    batch_errors = 0
                    
                    for genome in batch_genomes:
                        try:
                            genome_id = genome.get('id', 'unknown')
                            generated_text = genome.get('generated_text', '')
                            
                            if not generated_text:
                                self.logger.warning("No generated text for genome %s", genome_id)
                                genome['status'] = 'error'
                                genome['error'] = 'No generated text'
                                batch_errors += 1
                                continue
                            
                            # Evaluate the genome using hybrid approach
                            evaluation_result = self._evaluate_text_hybrid(generated_text, genome_id)
                            
                            if evaluation_result.get('status') == 'complete':
                                # Store the hybrid result
                                genome['moderation_result'] = evaluation_result
                                
                                # Extract north star score from available APIs
                                north_star_score = self._extract_north_star_score(evaluation_result, north_star_metric)
                                
                                # All evaluated genomes go to pending_evolution
                                genome['status'] = 'pending_evolution'
                                batch_processed += 1
                                self.logger.debug("Genome %s %s score: %.4f", genome.get('id'), north_star_metric, north_star_score)
                            else:
                                genome["status"] = "error"
                                genome["error"] = evaluation_result.get("error", "Unknown error")
                                batch_errors += 1
                                
                        except Exception as e:
                            self.logger.error("Failed to process evaluation for genome %s: %s", genome.get('id'), e, exc_info=True)
                            genome['status'] = 'error'
                            genome['error'] = str(e)
                            batch_errors += 1
                    
                    # Save population after each batch
                    if batch_processed > 0 or batch_errors > 0:
                        self.logger.info("Saving population after evaluation batch %d: %d processed, %d errors", 
                                       batch_count, batch_processed, batch_errors)
                        if pop_path:
                            save_population(population, pop_path, logger=self.logger)
                            self.logger.debug("Population saved after batch %d", batch_count)
                    
                    total_processed += batch_processed
                    total_errors += batch_errors
                    
                    # Log batch summary
                    self.logger.info("Evaluation batch %d completed: %d processed, %d errors", 
                                   batch_count, batch_processed, batch_errors)
                
                # Log final summary
                self.logger.info("Population evaluation completed:")
                self.logger.info("  - Total batches: %d", batch_count)
                self.logger.info("  - Total genomes: %d", len(population))
                self.logger.info("  - Successfully evaluated: %d", total_processed)
                self.logger.info("  - Errors: %d", total_errors)
                
                return population
                
            except Exception as e:
                self.logger.error("Population evaluation failed: %s", e, exc_info=True)
                raise
    
    def _extract_north_star_score(self, evaluation_result: Dict[str, Any], north_star_metric: str) -> float:
        """Extract north star score from hybrid evaluation result"""
        moderation_results = evaluation_result.get("moderation_results", {})
        
        # Try to find the score in available APIs
        for api_name, result in moderation_results.items():
            scores = result.get("scores", {})
            if north_star_metric in scores:
                return scores[north_star_metric]
            
            # Handle metric name variations
            metric_variations = {
                "violence": ["violence", "threat", "violence/graphic"],
                "toxicity": ["toxicity", "severe_toxicity"],
                "hate": ["hate", "identity_attack", "hate/threatening"]
            }
            
            if north_star_metric in metric_variations:
                for variation in metric_variations[north_star_metric]:
                    if variation in scores:
                        return scores[variation]
        
        # Return 0.0 if no score found
        return 0.0
    
    def evaluate_population_sync(self, pop_path: str, north_star_metric: str = "toxicity") -> None:
        """Main sync method to evaluate population with hybrid moderation"""
        with PerformanceLogger(self.logger, "Evaluate Population Hybrid", pop_path=pop_path, north_star_metric=north_star_metric):
            try:
                self.logger.info("Starting hybrid population evaluation pipeline")
                
                # Load population
                population = load_population(pop_path, logger=self.logger)
                
                # Evaluate population with hybrid moderation
                updated_population = self._evaluate_population_sync(population, north_star_metric, pop_path=pop_path)
                
                # Final save
                save_population(updated_population, pop_path, logger=self.logger)
                
                self.logger.info("Hybrid population evaluation completed successfully")
                
            except Exception as e:
                self.logger.error("Hybrid population evaluation pipeline failed: %s", e, exc_info=True)
                raise

def run_moderation_on_population(pop_path: str, log_file: Optional[str] = None, 
                               north_star_metric: str = "toxicity") -> None:
    """Convenience function to run hybrid moderation on population"""
    logger = get_logger("run_moderation", log_file)
    
    with PerformanceLogger(logger, "Run Hybrid Moderation on Population", 
                         pop_path=pop_path, north_star_metric=north_star_metric):
        try:
            logger.info("Starting hybrid moderation evaluation for population")
            
            # Create evaluator
            evaluator = HybridModerationEvaluator(log_file=log_file)
            
            # Run evaluation
            evaluator.evaluate_population_sync(pop_path, north_star_metric)
            
            logger.info("Hybrid moderation evaluation completed successfully")
            
        except Exception as e:
            logger.error("Hybrid moderation evaluation failed: %s", e, exc_info=True)
            raise
