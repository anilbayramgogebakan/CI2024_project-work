# Symbolic Regression

## Collobarators
Anil Bayram Gogebakan | s328470
[Oguzhan Akgun | s328919](https://github.com/oguzhaaan)
[Meric Ulucay | s328899](https://github.com/mericuluca/CI2024_project-work)

## Methodology
In this project, we implemented a symbolic regression algorithm. Our aim is to solve given dataset with the lowest MSE score. For this purpose, we use concepts that we learned in the class. In addition to that, we try to get some ideas from [“Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl”](https://arxiv.org/abs/2305.01582)  paper by Miles Cramer: . In this paper, symbolic regression algorithm is implemented as well. Specifically, tournament selection algorithm (with some modification), aging algorithm (killing the oldest individuals in the population)  and migration algorithm are the ones that we add to algorithms in addition to ones we learned in the class. Moreover, the paper gives a general idea about how we should organize our framework in general with some pseudo codes.
Firstly, we create a dataclass whose name is “Individual”. This dataclass represents an individual from a population. Thus, it has 5 features: Genome, fitness, fitness_val, age, and T value. Fitness and fitness_val are the MSE losses of individuals for train and validation splits respectively. Age and T values are for killing algorithm and simulated annealing algorithm respectively. Type of genome feature is “node” object which is also created by us. It is the tree-like representation of the mathematical formulas as implemented in genetic programming.
We create populations randomly where half of the population has lowest complexity, and the other half have the maximum complexity in given range by parameters. In every generation following algorithms applied in the given order after both fitness and fitness_val values assigned initially. Remember that every fitness value calculation, if there is a overflow (such as 0 in the denominator during divide operation), that individual is removed.

### Killing eldest individuals
In the beginning of the project, we were planning to remove individuals with the highest MSE score (higher fitness in our case). However, as mentioned in the PYSR paper, this approach will more useful in order to avoid from early convergence
### Tournament Selection
This method is also taken by PYSR paper. At first, we were planning to select top n individuals by sorting according to their fitness value. However, we realized that it also pushes our population to early convergence. However, tournament selection add some randomness and provides better training process. Moreover, it is implemented in the PYSR such that each tournament is done in parallel for efficiency concerns, but we didn’t implement parallelism because simply it was difficult.
### Crossover
Among tournament winners, we choose random two individual. Then we create two children by switching their subnodes
### Mutation
Among tournament winners, we applied mutation with simulated annealing
### Constant simplification
We realized that we have a lot of nodes with different operations, but these operations depend on some constant values rather than x (input) values. Thus, we calculate these nodes in order to reduce the complexity of individuals
### Elitism
In the PYSR paper, it is mentioned that elitism applied such that after elites are chosen from population, they are not included in the tournament. As a result of this, they are never exposed to neither mutation nor crossover. The idea behind this is also to avoid early convergence. However, we tried this but couldn’t get good results. Thus, we modified this approach as follows. After choosing elites, we keep their age as zero so that they are never eliminated from population. By doing so, we guarantee that if we find the best result, we will never eliminate them because of aging.
### Deduplication population
After some point, successful individuals with same genome might become very dominant. In this case, the population starts to lose its diversity. To overcome this, we decided to remove duplicates within the given interval.
### Simplify operation
We also realized that we have a lot of node with constant value if we try to calculate them such as “sin(0.3/0.5)”. Even though this kind of nodes provides variety, they also make individuals more complex than they are. Thus, we decided to simplify them by calculating in given intervals. It is important to note that we don’t do this every generation in order to benefit from variety. Thus, it makes this interval hyperparameter very important to make the balance between exploration and exploitation. Note that we only apply this simplification to the nodes whose are not depends on input x values.
### Killing constants
After some experiment, we realized that individuals whose nodes are only a constant becomes very dominant. Thus, we decide to eliminate them by removing every constant from population.
### Killing individuals with complex nodes
After some generations, we started to obtain very complex nodes. It was obvious that they are overfitting and push the whole population to become more complex since they won the tournament. In order to overcome this problem, we decided to remove complex individuals whose complexity value is above the given value. By doing so, we push algorithm to find better results with simple calculations.
Another thing worth mentioning is that we tried another algorithm to overcome the complexity problem. During the tournament phase, we decided to add some penalties according to individual’s complexity. Although we believe that it is a good approach, we couldn’t manage to optimize this multiplication constant (at least we believe this is the reason). Thus, we couldn’t get the results we desire and we decided not to use this approach.
### Early stopping
If existing best result has a cost lower than 0.0001, we stop searching for another solution.
### Limit population size
Even though hyper parameter config is used for balancing the population size (killing age, breeding number vice versa), sometimes population might become overcrowded. In these scenarios, hardware limitation problems might arise because of limited computational power. Thus, we decided to limit population size by removing individuals with worse fitness values.
### Migration
As mentioned in the PYSR paper, because of the nature of mutation and crossover, there might be a tendency to specific direction in population. From a perspective, it might be something bad since the population might stuck in local minimum. However, this might be something useful from another perspective. By generating more than one population, and exchanging tournament winners among each other, we might get closer to the absolute minimum. For instance, lets assume we have 4 population and each of the populations has tendency to different operations such as sin, cos, exp and log. If we apply migrating, we might have really good results with less iterations (generations). In the PYSR paper, parallelism is also used for evolving the populations at the same time. Also here, we didn’t implement parallelism since it was difficult to implement.
### Constant Optimization
After some experiments, we realized that the biggest weakness of our algorithm is the lack of constant optimization. In PYSR paper, BFGS method is mentioned but implementation wise, it seems very difficult. Thus, we came up with an alternative algorithm. First, we determined the constant in the individual’s nodes. Then, we applied mutation k times. At the end, we updated the constant if the loss is less than the existing one. As a very straight-forward solution, we couldn’t get what we want. This approach added a lot of computational loads with very few improvements in fitness. Thus, we decided not to use it.
### Extended operation simplification
After genomes become complex, we started to face situations where nodes can be simplified mathematically such as x*x-x*x. However, in symbolic regression, they might be very useful in case like when one of the x is mutated as y. Then equation becomes x*x-x*y which is a desired outcome. This situation is also mentioned in the PYSR paper and they decide not to simplify them but we still try to simplify these nodes in order to analyze it. However, not simplifying gives better results. Thus, we also didn’t use this algorithm.

By using these ideas and methodologies, we obtain good results in most of the given dataset. However, Dataset 2 and Dataset 8 were very challenging. Thus, we couldn’t able to generate good results for these tasks.

## Personal Contribution
I contributed this projecy by implementing following modules:
*   Individual
*   Crossover
*   Mutation
*   Tournament Selection
*   Killing eldest individuals
*   Killing constants
*   Elitism
*   Deduplication population
*   Killing complex nodes
*   Migration
*   Constant optimization
Also, I create the life cycle where we put all this methods in a order.