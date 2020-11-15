{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# One Max Problem"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Note: Run all the code cells below in order to ensure everything runs correctly"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The one max problem is a simple genetic algorithm problem. The objective is to find a bit string containing all 1s with a set length. We will use the functionality of the DEAP Python library to solve the problem."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First, we will import all the necessary modules."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import random\n",
    "\n",
    "from deap import base\n",
    "from deap import creator\n",
    "from deap import tools"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we need to define the fitness objective and individual classes using DEAP's Creator."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "creator.create(\"FitnessMax\", base.Fitness, weights=(1.0,))\n",
    "creator.create(\"Individual\", list, fitness=creator.FitnessMax)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the first line, we defined the name of the class, the inherited class, and the objectives. We inherited from DEAP's base.Fitness to create a class with all the functionality of an objective. Then, we defined a tuple, (1.0,). This tuple represents a single objective we want to maximize. If we wanted to minimize a single objective, we would replace the tuple with (-1.0,). Furthermore, if we wanted to create a multi-objective problem, we would replace the tuple with something like (1.0, 1.0)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the second line, we defined an \"Individual\" class as a list with a fitness defined as the objective above. You can think of the individual class as the individuals in our population. For the one max problem, we will define our bit string individuals as a list of Booleans represented as 1s and 0s."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will start defining the functions available to our genetic algorithm using DEAP's Toolbox."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "toolbox = base.Toolbox()\n",
    "# Attribute generator \n",
    "toolbox.register(\"attr_bool\", random.randint, 0, 1)\n",
    "# Structure initializers\n",
    "toolbox.register(\"individual\", tools.initRepeat, creator.Individual, \n",
    "    toolbox.attr_bool, 100)\n",
    "toolbox.register(\"population\", tools.initRepeat, list, toolbox.individual)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the code above, we started by defining our toolbox. Then, we added a random boolean generator and initializers for our individuals and population."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We defined \"attr_bool\" as a random generator producing either 0 or 1. Then, we defined \"individual\" as an individual generator. initRepeat takes in 3 arguments, a container (in this case a list), a function to fill the container, and how many times to call the function. This tells DEAP to initialize each individual with a list of 100 booleans essentially creating a bit string of length 100."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The final line defines our population as a list of individuals. We will later call the population function and tell it to produce a set number of individuals."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Documentation for creating types such as Fitness, Individuals, and Population can be found here: https://deap.readthedocs.io/en/master/tutorials/basic/part1.html"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we need to define our evaluation function for our fitness objective."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def evalOneMax(individual):\n",
    "    return sum(individual),"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We defined a simple evaluation function which returns the sum of the Boolean integers of an individual. This means individuals with more 1s will receive a higher fitness score with the maximum fitness score being 100. The sum is returned as a tuple to match the fitness objective we previously defined."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will define our genetic algorithm's genetic operators."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "toolbox.register(\"evaluate\", evalOneMax)\n",
    "toolbox.register(\"mate\", tools.cxTwoPoint)\n",
    "toolbox.register(\"mutate\", tools.mutFlipBit, indpb=0.05)\n",
    "toolbox.register(\"select\", tools.selTournament, tournsize=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the code above, we defined four functions for our genetic algorithm to use. Evaluate is defined as the evaluation function we previously defined. Mate is defined as a two-point crossover function. Mutate is defined as flipping a bit in our bitstring to either 1 or 0 respectively with an independent probability of flipping each individual bit of 5%."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Select is defined as a tournament selection of 3 individuals. Tournament selection is a common genetic operator where individuals are sampled and competed against each other. This selection process tends to preserve more varied traits than having the entire population compete against each other. There may be an individual in a sample with a valuable trait which is not in the top percentage of individuals in the entire population."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will start defining our main genetic algorithm."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def main():\n",
    "    pop = toolbox.population(n=300)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We initialized our population with 300 individuals in the line above. n is the parameter we left empty earlier when we defined our population."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will evaluate our population."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def main():\n",
    "    pop = toolbox.population(n=300)\n",
    "    \n",
    "    # Evaluate the entire population\n",
    "    fitnesses = list(map(toolbox.evaluate, pop))\n",
    "    for ind, fit in zip(pop, fitnesses):\n",
    "        ind.fitness.values = fit"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the code above, we mapped the evaluation function we defined earlier to our entire population. Then, we assigned each individual their respective fitness value."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will begin the evolutionary process."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def main():\n",
    "    pop = toolbox.population(n=300)\n",
    "    \n",
    "    # Evaluate the entire population\n",
    "    fitnesses = list(map(toolbox.evaluate, pop))\n",
    "    for ind, fit in zip(pop, fitnesses):\n",
    "        ind.fitness.values = fit\n",
    "        \n",
    "    # Begin the evolution\n",
    "    for g in range(40):\n",
    "        print(\"-- Generation %i --\" % g)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We start by defining the evolutionary loop and set the algorithm to run for 40 generations."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will add selection."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def main():\n",
    "    pop = toolbox.population(n=300)\n",
    "    \n",
    "    # Evaluate the entire population\n",
    "    fitnesses = list(map(toolbox.evaluate, pop))\n",
    "    for ind, fit in zip(pop, fitnesses):\n",
    "        ind.fitness.values = fit\n",
    "        \n",
    "    # Begin the evolution\n",
    "    for g in range(40):\n",
    "        print(\"-- Generation %i --\" % g)\n",
    "        \n",
    "        # Select the next generation individuals\n",
    "        offspring = toolbox.select(pop, len(pop))\n",
    "        # Clone the selected individuals\n",
    "        offspring = list(map(toolbox.clone, offspring))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the code above, we use tournament selection on the population and then make a list consisting of an exact copy of the selected individuals. This makes sure our offspring are a completely separate instance."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will perform crossover and mutation on the offspring."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def main():\n",
    "    pop = toolbox.population(n=300)\n",
    "    \n",
    "    # Evaluate the entire population\n",
    "    fitnesses = list(map(toolbox.evaluate, pop))\n",
    "    for ind, fit in zip(pop, fitnesses):\n",
    "        ind.fitness.values = fit\n",
    "        \n",
    "    # Begin the evolution\n",
    "    for g in range(40):\n",
    "        print(\"-- Generation %i --\" % g)\n",
    "        \n",
    "        # Select the next generation individuals\n",
    "        offspring = toolbox.select(pop, len(pop))\n",
    "        # Clone the selected individuals\n",
    "        offspring = list(map(toolbox.clone, offspring))\n",
    "\n",
    "        # Apply crossover and mutation on the offspring\n",
    "        for child1, child2 in zip(offspring[::2], offspring[1::2]):\n",
    "            if random.random() < 0.5:\n",
    "                toolbox.mate(child1, child2)\n",
    "                del child1.fitness.values\n",
    "                del child2.fitness.values\n",
    "\n",
    "        for mutant in offspring:\n",
    "            if random.random() < 0.2:\n",
    "                toolbox.mutate(mutant)\n",
    "                del mutant.fitness.values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The above code mates two individuals with a 50% probability and mutates an individual with a 20% probability using the mate and mutate functions we previously defined. The delete statements invalidate the fitness of the mated and mutated offspring. This is important for the next step."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will re-evaluate the modified offspring and replace the old population with the offspring."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def main():\n",
    "    pop = toolbox.population(n=300)\n",
    "    \n",
    "    # Evaluate the entire population\n",
    "    fitnesses = list(map(toolbox.evaluate, pop))\n",
    "    for ind, fit in zip(pop, fitnesses):\n",
    "        ind.fitness.values = fit\n",
    "        \n",
    "    # Begin the evolution\n",
    "    for g in range(40):\n",
    "        print(\"-- Generation %i --\" % g)\n",
    "        \n",
    "        # Select the next generation individuals\n",
    "        offspring = toolbox.select(pop, len(pop))\n",
    "        # Clone the selected individuals\n",
    "        offspring = list(map(toolbox.clone, offspring))\n",
    "\n",
    "        # Apply crossover and mutation on the offspring\n",
    "        for child1, child2 in zip(offspring[::2], offspring[1::2]):\n",
    "            if random.random() < 0.5:\n",
    "                toolbox.mate(child1, child2)\n",
    "                del child1.fitness.values\n",
    "                del child2.fitness.values\n",
    "\n",
    "        for mutant in offspring:\n",
    "            if random.random() < 0.2:\n",
    "                toolbox.mutate(mutant)\n",
    "                del mutant.fitness.values        \n",
    "    \n",
    "        # Evaluate the individuals with an invalid fitness\n",
    "        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]\n",
    "        fitnesses = map(toolbox.evaluate, invalid_ind)\n",
    "        for ind, fit in zip(invalid_ind, fitnesses):\n",
    "            ind.fitness.values = fit\n",
    "        \n",
    "        # Replace population\n",
    "        pop[:] = offspring"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finally, we will define statistics for our population and print them out."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def main():\n",
    "    pop = toolbox.population(n=300)\n",
    "    \n",
    "    # Evaluate the entire population\n",
    "    fitnesses = list(map(toolbox.evaluate, pop))\n",
    "    for ind, fit in zip(pop, fitnesses):\n",
    "        ind.fitness.values = fit\n",
    "        \n",
    "    # Begin the evolution\n",
    "    for g in range(40):\n",
    "        print(\"-- Generation %i --\" % g)\n",
    "        \n",
    "        # Select the next generation individuals\n",
    "        offspring = toolbox.select(pop, len(pop))\n",
    "        # Clone the selected individuals\n",
    "        offspring = list(map(toolbox.clone, offspring))\n",
    "\n",
    "        # Apply crossover and mutation on the offspring\n",
    "        for child1, child2 in zip(offspring[::2], offspring[1::2]):\n",
    "            if random.random() < 0.5:\n",
    "                toolbox.mate(child1, child2)\n",
    "                del child1.fitness.values\n",
    "                del child2.fitness.values\n",
    "\n",
    "        for mutant in offspring:\n",
    "            if random.random() < 0.2:\n",
    "                toolbox.mutate(mutant)\n",
    "                del mutant.fitness.values        \n",
    "    \n",
    "        # Evaluate the individuals with an invalid fitness\n",
    "        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]\n",
    "        fitnesses = map(toolbox.evaluate, invalid_ind)\n",
    "        for ind, fit in zip(invalid_ind, fitnesses):\n",
    "            ind.fitness.values = fit\n",
    "        \n",
    "        # Replace population\n",
    "        pop[:] = offspring\n",
    "    \n",
    "        # Gather all the fitnesses in one list and print the stats\n",
    "        fits = [ind.fitness.values[0] for ind in pop]\n",
    "        \n",
    "        length = len(pop)\n",
    "        mean = sum(fits) / length\n",
    "        sum2 = sum(x*x for x in fits)\n",
    "        std = abs(sum2 / length - mean**2)**0.5\n",
    "        \n",
    "        print(\"  Min %s\" % min(fits))\n",
    "        print(\"  Max %s\" % max(fits))\n",
    "        print(\"  Avg %s\" % mean)\n",
    "        print(\"  Std %s\" % std)\n",
    "        \n",
    "    print(\"-- End of (successful) evolution --\")\n",
    "    \n",
    "    best_ind = tools.selBest(pop, 1)[0]\n",
    "    print(\"Best individual is %s, %s\" % (best_ind, best_ind.fitness.values))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "main()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "I encourage you to run the code above several times. You might see the algorithm not always achieve the global maxmimum fitness in 40 generations. This is due to how our population was randomly initialized, the probability of crossover and mutation occuring, and the independent probability of flipping any bit when an individual is mutated."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You can think of the genetic algorithm we used as a search over the space of all possible bit strings of length 100. This approach is generally better than random search because we are optimizing our search space using our fitness objective. This concept will become important when we later talk about multi-objective problems."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Documentation for Operators and Algorithms can be found here: https://deap.readthedocs.io/en/master/tutorials/basic/part2.html"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You can find another notebook example of the One Max Problem here: https://github.com/DEAP/notebooks/blob/master/OneMax.ipynb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# The N Queens Problem"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, we're going to focus on experimenting with the genetic algorithm framework with a different problem."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The problem is to determine a configuration of n queens on a nxn chessboard such that no queen can be taken by one another. In this version, each queen is assigned to one column, and only one queen can be on each line."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This example can be found in the DEAP source code here: https://github.com/DEAP/deap/blob/master/examples/ga/nqueens.py"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First, we need to create our fitness and individual classes. We will also initialize a variable called n to store the size of our problem. Feel free to change this variable from the default value of n = 20."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Problem parameter\n",
    "n = 20\n",
    "\n",
    "creator.create(\"FitnessMin\", base.Fitness, weights=(-1.0,))\n",
    "creator.create(\"IndividualQueen\", list, fitness=creator.FitnessMin)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The main difference with our creation this time is the weight of the fitness objective. This time we will be minimizing an objective instead of maximizing an objective. This is because we want to minimize the number of conflicts between two queens on the chessboard."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will start defining our toolbox."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "#Since there is only one queen per line, \n",
    "#individual are represented by a permutation\n",
    "toolbox_q = base.Toolbox()\n",
    "toolbox_q.register(\"permutation\", random.sample, range(n), n)\n",
    "\n",
    "#Structure initializers\n",
    "#An individual is a list that represents the position of each queen.\n",
    "#Only the line is stored, the column is the index of the number in the list.\n",
    "toolbox_q.register(\"individual\", tools.initIterate, creator.IndividualQueen, toolbox_q.permutation)\n",
    "toolbox_q.register(\"population\", tools.initRepeat, list, toolbox_q.individual)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Similarly to the previous problem, we defined a function called \"permutation\" to help create our individuals and population. In this case, our individual becomes the return value of toolbox_q.permutation, which happens to be a list of integers sampled from range(n) without replacement."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "If n = 10, then an individual could look like this: [7,0,4,6,5,1,9,2,8,3]. If the first integer in the list is 7, then the queen on the first row will be in column 8, and if the second integer is 0, then the queen on the second row will be in column 1 and so on down the list."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"files/images/board_fancy.jpg\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The image above shows the solution for a N Queens problem where n = 8."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will define our evaluation function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def evalNQueens(individual):\n",
    "    size = len(individual)\n",
    "    #Count the number of conflicts with other queens.\n",
    "    #The conflicts can only be diagonal, count on each diagonal line\n",
    "    left_diagonal = [0] * (2*size-1)\n",
    "    right_diagonal = [0] * (2*size-1)\n",
    "    \n",
    "    #Sum the number of queens on each diagonal:\n",
    "    for i in range(size):\n",
    "        left_diagonal[i+individual[i]] += 1\n",
    "        right_diagonal[size-1-i+individual[i]] += 1\n",
    "    \n",
    "    #Count the number of conflicts on each diagonal\n",
    "    sum_ = 0\n",
    "    for i in range(2*size-1):\n",
    "        if left_diagonal[i] > 1:\n",
    "            sum_ += left_diagonal[i] - 1\n",
    "        if right_diagonal[i] > 1:\n",
    "            sum_ += right_diagonal[i] - 1\n",
    "    return sum_,"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The problem is to determine a configuration of n queens on a nxn chessboard such that no queen can be taken by one another. In this version, each queens is assigned to one column, and only one queen can be on each line. The evaluation function therefore only counts the number of conflicts along the diagonals."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "I encourage you to modify the above evaluation function. The evaluation function(s) defines your objective(s). If you tweak the evaluation function, then your domain will remain the same. However, the objective(s) you are trying to minimize or maximize in that domain will change."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will define our crossover function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def cxPartialyMatched(ind1, ind2):\n",
    "    \"\"\"Executes a partially matched crossover (PMX) on the input individuals.\n",
    "    The two individuals are modified in place. This crossover expects\n",
    "    :term:`sequence` individuals of indices, the result for any other type of\n",
    "    individuals is unpredictable.\n",
    "    \n",
    "    :param ind1: The first individual participating in the crossover.\n",
    "    :param ind2: The second individual participating in the crossover.\n",
    "    :returns: A tuple of two individuals.\n",
    "    Moreover, this crossover generates two children by matching\n",
    "    pairs of values in a certain range of the two parents and swapping the values\n",
    "    of those indexes. For more details see [Goldberg1985]_.\n",
    "    This function uses the :func:`~random.randint` function from the python base\n",
    "    :mod:`random` module.\n",
    "    \n",
    "    .. [Goldberg1985] Goldberg and Lingel, \"Alleles, loci, and the traveling\n",
    "       salesman problem\", 1985.\n",
    "    \"\"\"\n",
    "    size = min(len(ind1), len(ind2))\n",
    "    p1, p2 = [0]*size, [0]*size\n",
    "\n",
    "    # Initialize the position of each indices in the individuals\n",
    "    for i in range(size):\n",
    "        p1[ind1[i]] = i\n",
    "        p2[ind2[i]] = i\n",
    "    # Choose crossover points\n",
    "    cxpoint1 = random.randint(0, size)\n",
    "    cxpoint2 = random.randint(0, size - 1)\n",
    "    if cxpoint2 >= cxpoint1:\n",
    "        cxpoint2 += 1\n",
    "    else: # Swap the two cx points\n",
    "        cxpoint1, cxpoint2 = cxpoint2, cxpoint1\n",
    "    \n",
    "    # Apply crossover between cx points\n",
    "    for i in range(cxpoint1, cxpoint2):\n",
    "        # Keep track of the selected values\n",
    "        temp1 = ind1[i]\n",
    "        temp2 = ind2[i]\n",
    "        # Swap the matched value\n",
    "        ind1[i], ind1[p1[temp2]] = temp2, temp1\n",
    "        ind2[i], ind2[p2[temp1]] = temp1, temp2\n",
    "        # Position bookkeeping\n",
    "        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]\n",
    "        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]\n",
    "    \n",
    "    return ind1, ind2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For this problem we will be using partially matched crossover for mating our population. Partially matched crossover is specically used in this problem because it represents swapping around pairs of queen positions between two parent individuals. This should be more effective than swapping pieces of individuals around like in a one or two point crossover. If we swapped half of the chessboard we would not retain the information gained from either parent individual because the individual is formed row by row."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We have provided below an implementation of two point crossover below for you to test and experiment with."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def cxTwoPoint(ind1, ind2):\n",
    "    \"\"\"Executes a two-point crossover on the input :term:`sequence`\n",
    "    individuals. The two individuals are modified in place and both keep\n",
    "    their original length. \n",
    "    \n",
    "    :param ind1: The first individual participating in the crossover.\n",
    "    :param ind2: The second individual participating in the crossover.\n",
    "    :returns: A tuple of two individuals.\n",
    "    This function uses the :func:`~random.randint` function from the Python \n",
    "    base :mod:`random` module.\n",
    "    \"\"\"\n",
    "    size = min(len(ind1), len(ind2))\n",
    "    cxpoint1 = random.randint(1, size)\n",
    "    cxpoint2 = random.randint(1, size - 1)\n",
    "    if cxpoint2 >= cxpoint1:\n",
    "        cxpoint2 += 1\n",
    "    else: # Swap the two cx points\n",
    "        cxpoint1, cxpoint2 = cxpoint2, cxpoint1\n",
    "   \n",
    "    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \\\n",
    "        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]\n",
    "        \n",
    "    return ind1, ind2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will define our mutation function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def mutShuffleIndexes(individual, indpb):\n",
    "    \"\"\"Shuffle the attributes of the input individual and return the mutant.\n",
    "    The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the\n",
    "    probability of each attribute to be moved. Usually this mutation is applied on \n",
    "    vector of indices.\n",
    "    \n",
    "    :param individual: Individual to be mutated.\n",
    "    :param indpb: Independent probability for each attribute to be exchanged to\n",
    "                  another position.\n",
    "    :returns: A tuple of one individual.\n",
    "    \n",
    "    This function uses the :func:`~random.random` and :func:`~random.randint`\n",
    "    functions from the python base :mod:`random` module.\n",
    "    \"\"\"\n",
    "    size = len(individual)\n",
    "    for i in range(size):\n",
    "        if random.random() < indpb:\n",
    "            swap_indx = random.randint(0, size - 2)\n",
    "            if swap_indx >= i:\n",
    "                swap_indx += 1\n",
    "            individual[i], individual[swap_indx] = \\\n",
    "                individual[swap_indx], individual[i]\n",
    "    \n",
    "    return individual,"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "By default, we will use the shuffle indexes mutation for this problem. We want to shuffle the indexes around instead of changing them because the items in our individual lists represents position on a chessboard. Therefore, we cannot mutate any of the values to be duplicate or outside of the set bounds. However, there are many different ways to shuffle around the items in an individual's list."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Implement and test out at least one other mutation function in the code cell below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will register the functions we defined into our toolbox along with a tournament selection method."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "toolbox_q.register(\"evaluate\", evalNQueens)\n",
    "toolbox_q.register(\"mate\", cxPartialyMatched)\n",
    "toolbox_q.register(\"mutate\", mutShuffleIndexes, indpb=2.0/n)\n",
    "toolbox_q.register(\"select\", tools.selTournament, tournsize=3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, we can define and run our main evolutionary loop."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "pop = toolbox_q.population(n=300)\n",
    "\n",
    "# Evaluate the entire population\n",
    "fitnesses = list(map(toolbox_q.evaluate, pop))\n",
    "for ind, fit in zip(pop, fitnesses):\n",
    "    ind.fitness.values = fit\n",
    "\n",
    "# Begin the evolution\n",
    "for g in range(100):\n",
    "    print(\"-- Generation %i --\" % g)\n",
    "\n",
    "    # Select the next generation individuals\n",
    "    offspring = toolbox_q.select(pop, len(pop))\n",
    "    # Clone the selected individuals\n",
    "    offspring = list(map(toolbox_q.clone, offspring))\n",
    "\n",
    "    # Apply crossover and mutation on the offspring\n",
    "    for child1, child2 in zip(offspring[::2], offspring[1::2]):\n",
    "        if random.random() < 0.5:\n",
    "            toolbox_q.mate(child1, child2)\n",
    "            del child1.fitness.values\n",
    "            del child2.fitness.values\n",
    "\n",
    "    for mutant in offspring:\n",
    "        if random.random() < 0.2:\n",
    "            toolbox_q.mutate(mutant)\n",
    "            del mutant.fitness.values\n",
    "\n",
    "    # Evaluate the individuals with an invalid fitness\n",
    "    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]\n",
    "    fitnesses = map(toolbox_q.evaluate, invalid_ind)\n",
    "    for ind, fit in zip(invalid_ind, fitnesses):\n",
    "        ind.fitness.values = fit\n",
    "\n",
    "    # Replace population\n",
    "    pop[:] = offspring\n",
    "\n",
    "    # Gather all the fitnesses in one list and print the stats\n",
    "    fits = [ind.fitness.values[0] for ind in pop]\n",
    "\n",
    "    length = len(pop)\n",
    "    mean = sum(fits) / length\n",
    "    sum2 = sum(x*x for x in fits)\n",
    "    std = abs(sum2 / length - mean**2)**0.5\n",
    "\n",
    "    print(\"  Min %s\" % min(fits))\n",
    "    print(\"  Max %s\" % max(fits))\n",
    "    print(\"  Avg %s\" % mean)\n",
    "    print(\"  Std %s\" % std)\n",
    "\n",
    "print(\"-- End of (successful) evolution --\")\n",
    "\n",
    "best_ind = tools.selBest(pop, 1)[0]\n",
    "print(\"Best individual is %s, %s\" % (best_ind, best_ind.fitness.values))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Even after 100 generations, our algorithm is not guaranteed to end up with the global minimum of 0."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Tweak the parameters and functions of the N Queen Problem to find the global minimum with less iterations and with a higher consistency. You can reach those two objectives with separate parameters and functions."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "With our current main function we have to scroll through each generation and only see the list of the best individual at the end of evolution. Let's program a visualization to speed up our algorithm improvement process."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "gen = range(100)\n",
    "avg_list = []\n",
    "max_list = []\n",
    "min_list = []\n",
    "\n",
    "pop = toolbox_q.population(n=300)\n",
    "\n",
    "# Evaluate the entire population\n",
    "fitnesses = list(map(toolbox_q.evaluate, pop))\n",
    "for ind, fit in zip(pop, fitnesses):\n",
    "    ind.fitness.values = fit\n",
    "\n",
    "# Begin the evolution\n",
    "for g in gen:\n",
    "    print(\"-- Generation %i --\" % g)\n",
    "\n",
    "    # Select the next generation individuals\n",
    "    offspring = toolbox_q.select(pop, len(pop))\n",
    "    # Clone the selected individuals\n",
    "    offspring = list(map(toolbox_q.clone, offspring))\n",
    "\n",
    "    # Apply crossover and mutation on the offspring\n",
    "    for child1, child2 in zip(offspring[::2], offspring[1::2]):\n",
    "        if random.random() < 0.5:\n",
    "            toolbox_q.mate(child1, child2)\n",
    "            del child1.fitness.values\n",
    "            del child2.fitness.values\n",
    "\n",
    "    for mutant in offspring:\n",
    "        if random.random() < 0.2:\n",
    "            toolbox_q.mutate(mutant)\n",
    "            del mutant.fitness.values\n",
    "\n",
    "    # Evaluate the individuals with an invalid fitness\n",
    "    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]\n",
    "    fitnesses = map(toolbox_q.evaluate, invalid_ind)\n",
    "    for ind, fit in zip(invalid_ind, fitnesses):\n",
    "        ind.fitness.values = fit\n",
    "\n",
    "    # Replace population\n",
    "    pop[:] = offspring\n",
    "\n",
    "    # Gather all the fitnesses in one list and print the stats\n",
    "    fits = [ind.fitness.values[0] for ind in pop]\n",
    "\n",
    "    length = len(pop)\n",
    "    mean = sum(fits) / length\n",
    "    sum2 = sum(x*x for x in fits)\n",
    "    std = abs(sum2 / length - mean**2)**0.5\n",
    "    g_max = max(fits)\n",
    "    g_min = min(fits)\n",
    "        \n",
    "    avg_list.append(mean)\n",
    "    max_list.append(g_max)\n",
    "    min_list.append(g_min)\n",
    "\n",
    "    print(\"  Min %s\" % g_min)\n",
    "    print(\"  Max %s\" % g_max)\n",
    "    print(\"  Avg %s\" % mean)\n",
    "    print(\"  Std %s\" % std)\n",
    "\n",
    "print(\"-- End of (successful) evolution --\")\n",
    "\n",
    "best_ind = tools.selBest(pop, 1)[0]\n",
    "print(\"Best individual is %s, %s\" % (best_ind, best_ind.fitness.values))\n",
    "    \n",
    "import matplotlib.pyplot as plt\n",
    "plt.plot(gen, avg_list, label=\"average\")\n",
    "plt.plot(gen, min_list, label=\"minimum\")\n",
    "plt.plot(gen, max_list, label=\"maximum\")\n",
    "plt.xlabel(\"Generation\")\n",
    "plt.ylabel(\"Fitness\")\n",
    "plt.legend(loc=\"upper right\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After running the code cells above, you should see a plot of the average, minimum, and maximum over 100 generations. We added lists to store the average, minimum, and maximum of each generation and then used matplotlib to plot all of the data. Plots are useful because they show us useful information about the data we collected. In the plot above, we can see the average and minimum reach a stable bend at around 20-30 generations in. Now we have a visualization of one of the optimization problems we instructed you to solve."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
