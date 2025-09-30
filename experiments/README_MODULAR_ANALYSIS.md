# 📚 Modular Evolution Experiment Analysis

**A Comprehensive, Modular Analysis Framework for Evolutionary Text Generation**

---

## 🎯 Overview

The evolution experiment analysis has been divided into **9 focused, modular notebooks** for better organization, maintainability, and flexibility. Each notebook handles a specific aspect of the analysis pipeline.

## 📁 Notebook Structure

### 🎛️ **Master Control**
- **`00_master_analysis.ipynb`** - Main orchestration notebook that runs all sections

### 🔧 **Core Modules**

1. **`01_setup_configuration.ipynb`** - 🔧 Setup & Configuration
   - Environment setup and imports
   - Professional visualization settings
   - Core utility functions
   - Data schema definitions

2. **`02_data_loading.ipynb`** - 📂 Data Loading & Validation
   - Population data loading with fallback paths
   - Comprehensive schema validation
   - Data quality assessment
   - Toxicity analysis setup

3. **`03_operator_analysis.ipynb`** - ⚙️ Operator Analysis
   - Operator performance analysis (16 operators: 13 mutation + 3 crossover)
   - Usage statistics and visualizations
   - Performance tier classification
   - Generation-wise operator analysis
   - Multi-language back-translation analysis
   - Dual translation approach comparison

4. **`04_data_quality.ipynb`** - 🧹 Data Quality Analysis
   - Detailed data inspection
   - Duplicate prompt detection
   - Data cleaning and preprocessing
   - Quality metrics and reports

5. **`05_lexical_diversity.ipynb`** - 📝 Lexical Diversity Analysis
   - Text processing and tokenization
   - Type-token ratio analysis
   - Shannon entropy calculations
   - Lexical diversity visualizations

6. **`06_evolution_progress.ipynb`** - 📈 Evolution Progress Analysis
   - EvolutionTracker.json analysis
   - Generation progress tracking
   - Score trajectory analysis
   - Evolution insights

7. **`07_performance_dashboard.ipynb`** - 📊 Performance Dashboard
   - Multi-panel performance visualizations
   - Comprehensive dashboards
   - Interactive performance metrics
   - Summary statistics

8. **`08_semantic_analysis.ipynb`** - 🧠 Semantic Analysis
   - LLaMA embeddings generation
   - Semantic similarity calculations
   - Parent-child drift analysis
   - Advanced semantic visualizations

9. **`09_reporting_export.ipynb`** - 📑 Reporting & Export
   - Comprehensive report generation
   - LaTeX table exports
   - Publication-ready materials
   - Data export utilities

---

## 🚀 Usage Options

### Option 1: Complete Pipeline (Recommended)
```bash
# Run the master notebook for complete analysis
jupyter notebook 00_master_analysis.ipynb
```

### Option 2: Individual Sections
```bash
# Run specific analysis sections as needed
jupyter notebook 03_operator_analysis.ipynb
jupyter notebook 08_semantic_analysis.ipynb
```

### Option 3: Selective Execution
```python
# In the master notebook, customize which sections to run
SECTIONS_TO_RUN = {
    'setup_config': True,
    'data_loading': True,
    'operator_analysis': True,      # Focus on operator analysis
    'semantic_analysis': False,     # Skip heavy semantic analysis
    # ... customize as needed
}
```

---

## 🎯 Benefits of Modular Structure

### ✅ **Advantages**
- **🔧 Maintainability**: Each section is self-contained and easier to debug
- **🚀 Performance**: Run only the sections you need
- **🔄 Reusability**: Individual notebooks can be reused across projects
- **👥 Collaboration**: Team members can work on different sections independently
- **🎛️ Flexibility**: Mix and match sections based on analysis needs
- **📚 Organization**: Clear separation of concerns and logical flow

### 📊 **Use Cases**
- **Full Analysis**: Run master notebook for complete pipeline
- **Quick Operator Check**: Run just `03_operator_analysis.ipynb`
- **Data Quality Focus**: Run `02_data_loading.ipynb` + `04_data_quality.ipynb`
- **Publication Prep**: Run `08_semantic_analysis.ipynb` + `09_reporting_export.ipynb`

---

## 🔗 Dependencies

### **Notebook Dependencies**
- All notebooks depend on `01_setup_configuration.ipynb` for core setup
- Some notebooks may depend on previous data processing steps
- Dependencies are clearly marked at the top of each notebook

### **Data Dependencies**
- Primary: `../outputs/elites.json` (steady-state population)
- Secondary: `../outputs/EvolutionTracker.json`
- Optional: `../outputs/Population.json` (full population if needed)
- All notebooks include intelligent path resolution

---

## 📈 Migration from Original

### **Original Notebook**
- `experiments.ipynb` - Single large notebook (2500+ lines)

### **New Structure**
- 9 focused notebooks (~200-400 lines each)
- Better error isolation and debugging
- Faster execution of individual sections

### **Backward Compatibility**
- Original `experiments.ipynb` remains available
- Same analysis capabilities, just better organized
- All outputs remain in the same `experiments/` directory

---

## 🛠️ Customization

### **Section Control**
Each notebook can be run independently or skipped in the master notebook:

```python
# Customize in 00_master_analysis.ipynb
SECTIONS_TO_RUN = {
    'operator_analysis': True,      # Always run
    'semantic_analysis': False,     # Skip if no GPU
    'reporting_export': True        # Run for publications
}
```

### **Output Control**
- All outputs saved to `experiments/` directory
- Timestamped files for easy tracking
- Consistent naming conventions across modules

---

## 📝 Next Steps

1. **Start with Master**: Run `00_master_analysis.ipynb` for complete analysis
2. **Explore Sections**: Dive into individual notebooks for specific analysis
3. **Customize**: Modify section selection based on your needs
4. **Extend**: Add new analysis sections following the same pattern

---

**Happy Analyzing! 🎉** 