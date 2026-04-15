# Risk Engine & Placement Optimizer

The `risk_engine` is responsible for finding the optimal placement of EV charging infrastructure to minimize cumulative grid risk.

## Components

- **`optimizer_ga.py`**: A **Genetic Algorithm (GA)** based optimizer. It evaluates thousands of placement candidates against generated "extreme" demand scenarios.
- **Fitness Function**: Combines Gini-coefficient fairness with grid reliability metrics (Expected Energy Not Served).
- **Hard Constraints**: Ensures that any proposed placement remains under the total transformer capacity of the feeder.

## Usage

The optimizer is typically invoked from the root via:
```bash
python run.py optimize --scenarios 10 --iterations 50
```
It outputs a JSON recommendation (`final_optimal_layout.json`) mapping charger counts to specific IEEE-33 bus nodes.
