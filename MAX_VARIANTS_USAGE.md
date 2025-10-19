# Max Variants Argument Usage

The `--max-variants` argument controls how many variants each operator generates per call.

## Usage Examples

### Basic Usage
```bash
python3 src/main.py --max-variants 1
```

### Generate More Variants
```bash
python3 src/main.py --max-variants 5
```

### Combined with Other Arguments
```bash
python3 src/main.py --max-variants 3 --operators ie --threshold 0.95
```

## How It Works

- `max_variants` controls how many variants each operator generates per call
- Each operator is called `max_variants` times, and each call generates 1 variant
- Higher values will generate more variants per operator but take longer to complete
- Default value is 1 (single variant per operator call)

## Parameter Details

- **Type**: Integer
- **Default**: 1
- **Range**: 1 or higher
- **Effect**: Multiplies the number of variants generated per operator

## Operator Mode Behavior

- **IE mode** (`--operators ie`): Only InformedEvolution operator runs, no other operators are initialized
- **CM mode** (`--operators cm`): All operators except InformedEvolution run
- **All mode** (`--operators all`): All operators run (default)
