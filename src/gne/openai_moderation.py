import os
import json
import asyncio
import hashlib
import time
from typing import List, Dict, Optional, Tuple, Any
import openai
from openai import OpenAI as OpenAIClient, AsyncOpenAI
from dotenv import load_dotenv
from utils.custom_logging import get_logger, get_log_filename, PerformanceLogger
from concurrent.futures import ThreadPoolExecutor
import threading
import aiohttp
from utils.population_io import load_population, save_population

# Load environment variables
load_dotenv()

# Placeholder for module-level logger. It will be initialised via get_logger
logger = get_logger("openai_moderation", get_log_filename())

# Simple in-memory cache for moderation results
_moderation_cache = {}
_cache_lock = threading.Lock()

# Global thread pool for async operations
_thread_pool = None

def _get_thread_pool():
    """Get or create thread pool for async operations"""
    global _thread_pool
    if _thread_pool is None:
        # Test different values:
        # 4 = Conservative (current)
        # 8 = Moderate (recommended)
        # 16 = Aggressive (matches CPU cores)
        # 32 = Very aggressive (2x CPU cores)
        _thread_pool = ThreadPoolExecutor(max_workers=8)  # Changed from 4 to 8
    return _thread_pool

def _cleanup_thread_pool():
    """Clean up thread pool to prevent resource leaks"""
    global _thread_pool
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=True)
        _thread_pool = None

def _get_text_hash(text: str) -> str:
    """Generate a hash for text to use as cache key"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def _get_cached_result(text: str) -> Optional[Dict]:
    """Get cached moderation result if available"""
    text_hash = _get_text_hash(text)
    with _cache_lock:
        return _moderation_cache.get(text_hash)

def _cache_result(text: str, result: Dict):
    """Cache moderation result"""
    text_hash = _get_text_hash(text)
    with _cache_lock:
        _moderation_cache[text_hash] = result

async def evaluate_moderation_async(client: AsyncOpenAI, texts: List[str]) -> List[Dict]:
    """Async batch evaluation of multiple texts"""
    if not texts:
        return []
    
    # Check cache first
    cached_results = []
    uncached_texts = []
    uncached_indices = []
    
    for i, text in enumerate(texts):
        cached = _get_cached_result(text)
        if cached:
            cached_results.append((i, cached))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    results = [None] * len(texts)
    
    # Fill in cached results
    for idx, result in cached_results:
        results[idx] = result
    
    # Process uncached texts in batches (OpenAI allows up to 100 inputs per request)
    batch_size = 100
    for batch_start in range(0, len(uncached_texts), batch_size):
        batch_texts = uncached_texts[batch_start:batch_start + batch_size]
        batch_indices = uncached_indices[batch_start:batch_start + batch_size]
        
        try:
            response = await client.moderations.create(
                model="omni-moderation-latest",
                input=batch_texts
            )
            
            batch_results = response.to_dict()
            
            # Process batch results
            for i, (text, result_idx) in enumerate(zip(batch_texts, batch_indices)):
                if i < len(batch_results["results"]):
                    result = {
                        "id": batch_results["id"],
                        "model": batch_results["model"],
                        "results": [batch_results["results"][i]]
                    }
                    results[result_idx] = result
                    _cache_result(text, result)  # Cache for future use
        except Exception as e:
            logger.error(f"Batch moderation API call failed: {e}")
            # Fill failed results with None
            for result_idx in batch_indices:
                if results[result_idx] is None:
                    results[result_idx] = None
    
    return results

def evaluate_moderation(text: str) -> Optional[Dict]:
    """Single text evaluation (backwards compatibility)"""
    # Check cache first
    cached = _get_cached_result(text)
    if cached:
        return cached
    
    try:
        client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORG_ID"),
            project=os.getenv("OPENAI_PROJECT_ID")
        )
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=text
        )
        result = response.to_dict()
        _cache_result(text, result)  # Cache for future use
        logger.debug(f"Moderation response: {response}")
        return result
    except Exception as e:
        logger.error(f"Moderation API call failed: {e}")
        return None

def compute_toxicity_score(score_dict: Dict) -> float:
    """Compute average of category scores to use as toxicity score."""
    if not score_dict:
        return 0.0
    scores = list(score_dict.values())
    return sum(scores) / len(scores)

async def run_moderation_batch_async(texts: List[str]) -> List[Optional[Dict]]:
    """Run moderation on a batch of texts asynchronously"""
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG_ID"),
        project=os.getenv("OPENAI_PROJECT_ID")
    )
    
    return await evaluate_moderation_async(client, texts)

def normalize_moderation_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize moderation scores to be between 0.0001 and 1.0000 with 4 decimal places.
    
    Args:
        scores: Raw scores from OpenAI moderation API
        
    Returns:
        Normalized scores between 0.0001 and 1.0000
    """
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

class OpenAIModerationEvaluator:
    """Optimized OpenAI moderation evaluator for evolutionary text generation"""
    
    def __init__(self, log_file: Optional[str] = None, config_path: str = "config/modelConfig.yaml"):
        """Initialize the OpenAI moderation evaluator with logging and config"""
        self.logger = get_logger("OpenAIModerationEvaluator", log_file)
        self.logger.info("Initializing OpenAI Moderation Evaluator")

        # Load model config for batch size settings
        try:
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            # Use the first model's config (assuming single model setup)
            model_key = list(config.keys())[0] if config else "llama"
            self.model_cfg = config.get(model_key, {})
            self.logger.info("Model config loaded for batch size configuration")
        except Exception as e:
            self.logger.warning("Failed to load model config: %s, using defaults", e)
            self.model_cfg = {}

        # API configuration
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.org_id = os.getenv("OPENAI_ORG_ID")
        self.project_id = os.getenv("OPENAI_PROJECT_ID")
        
        if not self.api_key:
            self.logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.logger.info("OpenAI API configuration loaded")
        self.logger.debug("Organization ID: %s", self.org_id)
        self.logger.debug("Project ID: %s", self.project_id)
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        
        # API endpoints
        self.base_url = "https://api.openai.com/v1"
        self.moderation_url = f"{self.base_url}/moderations"
        
        self.logger.debug("OpenAI Moderation Evaluator initialized successfully")
    
    def _load_population(self, pop_path: str) -> List[Dict[str, Any]]:
        """Wrapper around utils.population_io.load_population for backward compatibility."""
        return load_population(pop_path, logger=self.logger)
    
    def _save_population(self, population: List[Dict[str, Any]], pop_path: str) -> None:
        save_population(population, pop_path, logger=self.logger)
    
    async def _evaluate_text_async(self, text: str, genome_id: str) -> Dict[str, Any]:
        """Evaluate a single text asynchronously with detailed logging"""
        with PerformanceLogger(self.logger, "Evaluate Text Async", genome_id=genome_id, text_length=len(text)):
            try:
                self.logger.debug("Evaluating text for genome %s: %d characters", genome_id, len(text))
                
                # Prepare request payload
                payload = {
                    "input": text,
                    "model": "text-moderation-latest"
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                if self.org_id:
                    headers["OpenAI-Organization"] = self.org_id
                
                # Make API request
                async with aiohttp.ClientSession() as session:
                    start_time = time.time()
                    
                    async with session.post(
                        self.moderation_url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            result = await response.json()
                            self.logger.debug("API response received for genome %s in %.3f seconds", 
                                            genome_id, response_time)
                            
                            # Process moderation result
                            moderation_result = self._process_moderation_result(result, genome_id)
                            
                            # Update performance metrics
                            self.evaluation_count += 1
                            self.successful_evaluations += 1
                            self.total_evaluation_time += response_time
                            
                            return moderation_result
                        else:
                            # Non-200 response handling to avoid returning None
                            try:
                                error_text = await response.text()
                            except Exception:
                                error_text = f"HTTP {response.status} with unreadable body"
                            self.logger.error(
                                "Moderation API returned non-200 for genome %s: %s - %s",
                                genome_id,
                                response.status,
                                error_text,
                            )
                            self.evaluation_count += 1
                            self.failed_evaluations += 1
                            return {
                                "genome_id": genome_id,
                                "status": "error",
                                "error": f"HTTP {response.status}: {error_text}",
                                "evaluation_timestamp": time.time(),
                            }
                            
            except Exception as e:
                self.logger.error("Unexpected error evaluating genome %s: %s", genome_id, e, exc_info=True)
                self.evaluation_count += 1
                self.failed_evaluations += 1
                return {
                    "genome_id": genome_id,
                    "status": "error",
                    "error": str(e),
                    "evaluation_timestamp": time.time()
                }
    
    def _process_moderation_result(self, api_result: Dict[str, Any], genome_id: str) -> Dict[str, Any]:
        """Process OpenAI moderation API result with detailed logging"""
        with PerformanceLogger(self.logger, "Process Moderation Result", genome_id=genome_id):
            try:
                self.logger.debug("Processing moderation result for genome %s", genome_id)
                
                # Extract results from API response
                results = api_result.get("results", [])
                if not results:
                    self.logger.warning("No results found in API response for genome %s", genome_id)
                    return {
                        "genome_id": genome_id,
                        "status": "error",
                        "error": "No results in API response",
                        "evaluation_timestamp": time.time()
                    }
                
                result = results[0]
                flagged = result.get("flagged", False)
                raw_categories = result.get("category_scores", {})
                
                # Normalize scores to 0.0001-1.0000 range with 4 decimal places
                normalized_categories = normalize_moderation_scores(raw_categories)
                
                # Log detailed scores (before and after normalization)
                self.logger.debug("Raw moderation scores for genome %s:", genome_id)
                for category, score in raw_categories.items():
                    self.logger.debug("  %s: %.6f", category, score)
                
                self.logger.debug("Normalized moderation scores for genome %s:", genome_id)
                for category, score in normalized_categories.items():
                    self.logger.debug("  %s: %.4f", category, score)
                
                # Create moderation result with normalized scores
                moderation_result = {
                    "genome_id": genome_id,
                    "status": "complete",
                    "flagged": flagged,
                    "scores": normalized_categories,
                    "evaluation_timestamp": time.time()
                }
                
                # Log summary
                if flagged:
                    self.logger.warning("Genome %s flagged for moderation", genome_id)
                    flagged_categories = [cat for cat, score in normalized_categories.items() if score > 0.5]
                    self.logger.warning("Flagged categories: %s", flagged_categories)
                else:
                    self.logger.info("Genome %s passed moderation", genome_id)
                
                return moderation_result
                
            except Exception as e:
                self.logger.error("Failed to process moderation result for genome %s: %s", genome_id, e, exc_info=True)
                return {
                    "genome_id": genome_id,
                    "status": "error",
                    "error": f"Failed to process result: {str(e)}",
                    "evaluation_timestamp": time.time()
                }
    
    async def _evaluate_population_async(self, population: List[Dict[str, Any]], 
                                       north_star_metric: str = "violence", batch_size: int = None, pop_path: str = "") -> List[Dict[str, Any]]:
        """Evaluate entire population asynchronously with batch saving for fault tolerance"""
        # Use config batch size if not provided, fallback to default
        if batch_size is None:
            batch_size = self.model_cfg.get("evaluation_batch_size", 10)
        
        with PerformanceLogger(self.logger, "Evaluate Population Async", 
                             population_size=len(population), north_star_metric=north_star_metric, batch_size=batch_size):
            try:
                self.logger.info("Starting async population evaluation with batch saving")
                self.logger.info("North star metric: %s", north_star_metric)
                self.logger.info("Using batch size: %d (from config: %s)", batch_size, 
                               self.model_cfg.get("evaluation_batch_size", "default"))
                
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
                
                toxic_prompt_ids = set()
                most_toxic_genomes = []
                
                for i in range(0, len(pending_genomes), batch_size):
                    batch_count += 1
                    batch_end = min(i + batch_size, len(pending_genomes))
                    batch_genomes = pending_genomes[i:batch_end]
                    
                    self.logger.info("Processing evaluation batch %d: genomes %d-%d", 
                                   batch_count, i + 1, batch_end)
                    
                    # Create evaluation tasks for this batch
                    tasks = []
                    for genome in batch_genomes:
                        genome_id = genome.get('id', 'unknown')
                        generated_text = genome.get('generated_text', '')
                        
                        if not generated_text:
                            self.logger.warning("No generated text for genome %s", genome_id)
                            genome['status'] = 'error'
                            genome['error'] = 'No generated text'
                            continue
                        
                        task = self._evaluate_text_async(generated_text, genome_id)
                        tasks.append((genome, task))
                    
                    self.logger.info("Created %d evaluation tasks for batch %d", len(tasks), batch_count)
                    
                    if not tasks:
                        self.logger.info("No valid tasks in batch %d, skipping", batch_count)
                        continue
                    
                    # Execute tasks with concurrency limit
                    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
                    
                    async def evaluate_with_semaphore(genome, task):
                        async with semaphore:
                            return await task
                    
                    # Process results for this batch
                    batch_processed = 0
                    batch_errors = 0
                    
                    for genome, task in tasks:
                        try:
                            evaluation_result = await evaluate_with_semaphore(genome, task)
                            # Guard against None results
                            if not evaluation_result:
                                genome["status"] = "error"
                                genome["error"] = "No result returned from moderation API"
                                batch_errors += 1
                                continue
                            if evaluation_result.get('status') == 'complete':
                                genome['moderation_result'] = evaluation_result
                                north_star_score = evaluation_result.get('scores', {}).get(north_star_metric, 0)
                                # All evaluated genomes go to pending_evolution regardless of score
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
                        # Save the full population after each batch for fault tolerance
                        if pop_path:
                            self._save_population(population, pop_path)
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
                self.logger.info("  - Skipped: %d", len(population) - total_processed - total_errors)
                
                # Return population without special most_toxic handling
                return population
                
            except Exception as e:
                self.logger.error("Population evaluation failed: %s", e, exc_info=True)
                raise
    
    async def evaluate_population_async(self, pop_path: str, north_star_metric: str = "violence") -> None:
        """Main async method to evaluate population with comprehensive logging"""
        with PerformanceLogger(self.logger, "Evaluate Population", pop_path=pop_path, north_star_metric=north_star_metric):
            try:
                self.logger.info("Starting population evaluation pipeline")
                
                # Load population
                population = self._load_population(pop_path)
                
                # Evaluate population with batch saving
                updated_population = await self._evaluate_population_async(population, north_star_metric, pop_path=pop_path)
                
                # Final save (in case there were any remaining changes)
                self._save_population(updated_population, pop_path)
                
                # Log performance metrics
                if self.evaluation_count > 0:
                    success_rate = (self.successful_evaluations / self.evaluation_count) * 100
                    avg_time = self.total_evaluation_time / self.successful_evaluations if self.successful_evaluations > 0 else 0
                    
                    self.logger.info("Evaluation Performance:")
                    self.logger.info("  - Total evaluations: %d", self.evaluation_count)
                    self.logger.info("  - Successful: %d (%.1f%%)", self.successful_evaluations, success_rate)
                    self.logger.info("  - Failed: %d", self.failed_evaluations)
                    self.logger.info("  - Average time per evaluation: %.3f seconds", avg_time)
                
            except Exception as e:
                self.logger.error("Population evaluation pipeline failed: %s", e, exc_info=True)
                raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the evaluator"""
        stats = {
            'evaluation_count': self.evaluation_count,
            'successful_evaluations': self.successful_evaluations,
            'failed_evaluations': self.failed_evaluations,
            'total_evaluation_time': self.total_evaluation_time
        }
        
        if self.evaluation_count > 0:
            stats['success_rate'] = (self.successful_evaluations / self.evaluation_count) * 100
            stats['failure_rate'] = (self.failed_evaluations / self.evaluation_count) * 100
        
        if self.successful_evaluations > 0:
            stats['average_evaluation_time'] = self.total_evaluation_time / self.successful_evaluations
        
        self.logger.debug("Performance stats: %s", stats)
        return stats

def run_moderation_on_population(pop_path: str, log_file: Optional[str] = None, 
                               north_star_metric: str = "violence") -> None:
    """Convenience function to run moderation on population with comprehensive logging"""
    logger = get_logger("run_moderation", log_file)
    
    with PerformanceLogger(logger, "Run Moderation on Population", 
                         pop_path=pop_path, north_star_metric=north_star_metric):
        try:
            logger.info("Starting moderation evaluation for population")
            
            # Create evaluator
            evaluator = OpenAIModerationEvaluator(log_file=log_file)
            
            # Run evaluation
            asyncio.run(evaluator.evaluate_population_async(pop_path, north_star_metric))
            
            # Update EvolutionTracker with best genomes after evaluation
            update_evolution_tracker_after_evaluation(pop_path, north_star_metric, logger)
            
            # Log final statistics
            stats = evaluator.get_performance_stats()
            logger.info("Moderation evaluation completed successfully")
            logger.info("Final statistics: %s", stats)
            
        except Exception as e:
            logger.error("Moderation evaluation failed: %s", e, exc_info=True)
            raise

def update_evolution_tracker_after_evaluation(pop_path: str, north_star_metric: str, logger):
    """Update EvolutionTracker with best genome for each generation after evaluation"""
    from pathlib import Path
    import json
    from ea.RunEvolution import evolution_tracker_path
    
    try:
        # Load population
        population = load_population(pop_path, logger=logger)
        
        # Load EvolutionTracker
        if not evolution_tracker_path.exists():
            logger.warning("EvolutionTracker not found, skipping update")
            return
            
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            evolution_tracker = json.load(f)
        
        # Group genomes by prompt_id and generation
        genomes_by_prompt_gen = {}
        for genome in population:
            prompt_id = genome.get("prompt_id")
            generation = genome.get("generation")
            if prompt_id is not None and generation is not None:
                key = (prompt_id, generation)
                if key not in genomes_by_prompt_gen:
                    genomes_by_prompt_gen[key] = []
                genomes_by_prompt_gen[key].append(genome)
        
        # Update EvolutionTracker with best genomes
        updated_count = 0
        for entry in evolution_tracker:
            prompt_id = entry["prompt_id"]
            
            for gen in entry["generations"]:
                gen_number = gen["generation_number"]
                key = (prompt_id, gen_number)
                
                if key in genomes_by_prompt_gen:
                    # Find best genome for this generation
                    genomes = genomes_by_prompt_gen[key]
                    best_genome = None
                    best_score = 0.0
                    
                    for genome in genomes:
                        if genome.get("moderation_result"):
                            score = (genome["moderation_result"].get("scores", {}).get(north_star_metric, 0.0))
                            if score > best_score:
                                best_score = score
                                best_genome = genome
                    
                    if best_genome:
                        # Update EvolutionTracker with best genome info
                        gen["genome_id"] = best_genome["id"]
                        gen["max_score"] = best_score
                        updated_count += 1
                        logger.debug(f"Updated generation {gen_number} for prompt_id {prompt_id}: genome_id={best_genome['id']}, score={best_score}")
        
        # Save updated EvolutionTracker
        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Updated EvolutionTracker with {updated_count} best genomes after evaluation")
        
    except Exception as e:
        logger.error(f"Failed to update EvolutionTracker after evaluation: {e}", exc_info=True)
        # Don't raise - this is not critical for the main evaluation process

# Batch processing utilities for efficient moderation
def batch_moderate_texts(texts: List[str], batch_size: int = 100) -> List[Optional[Dict]]:
    """Synchronous batch moderation with optimal batch sizes"""
    if not texts:
        return []
    
    logger.info(f"Batch moderating {len(texts)} texts in batches of {batch_size}")
    
    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG_ID"),
        project=os.getenv("OPENAI_PROJECT_ID")
    )
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        try:
            response = client.moderations.create(
                model="omni-moderation-latest",
                input=batch_texts
            )
            
            batch_results = response.to_dict()
            
            # Process each result in the batch
            for j, text in enumerate(batch_texts):
                if j < len(batch_results["results"]):
                    result = {
                        "id": batch_results["id"],
                        "model": batch_results["model"],
                        "results": [batch_results["results"][j]]
                    }
                    results.append(result)
                    _cache_result(text, result)  # Cache for future use
                else:
                    results.append(None)
                    
        except Exception as e:
            logger.error(f"Batch moderation failed for batch starting at {i}: {e}")
            # Add None results for failed batch
            results.extend([None] * len(batch_texts))
    
    return results

def clear_moderation_cache():
    """Clear the moderation cache"""
    global _moderation_cache
    with _cache_lock:
        _moderation_cache.clear()
    logger.info("Moderation cache cleared")

def get_cache_stats() -> Dict:
    """Get cache statistics"""
    with _cache_lock:
        return {
            "cache_size": len(_moderation_cache),
            "cache_keys": list(_moderation_cache.keys())[:10]  # First 10 keys for debugging
        }