#Instructions

###Run your own data
1. Specifying type for each attributes is required.
2. Specifying selected splitting attributes is required.
3. Create the instance of a `DecisionTree`
4. Load data supports from file or from `ArrayList<Sting[]>` type data. This step is required for next step.
5. After loading the data, call `startTraining()` method. This step is required for `startTesting()` method.
6. (Optional) After step 5, call `startTesting()`.

###Run examples from main
Run steps:
1. Load two java files from src;
2. Run main.java.

Notice:

1. The dataset "smallerData.csv" now contains both categorical and numerical data.

The output includes a tree diagram and its prediction confusion matrix.

Updates:

For the categorical feature, it finds the minEntropy feature and corresponding feature value, making an EQUAL and NOT EQUAL branch.

This post explains well about this issue.
http://stats.stackexchange.com/questions/12187/are-decision-trees-almost-always-binary-trees

For numerical features, treating every numerical value of that attribute as a decision boundary, then choose the minEntropy attribute and the corresponding feature value.

This version also allows user specified feature choosing(ignoring), a user can decide what feature is redundant and not considering them when splitting.

![Screen Shot 2017-03-02 at 20.41.32.png](https://svbtleusercontent.com/phldkfvsbzstoq.png)

![Screen Shot 2017-03-02 at 20.41.35.png](https://svbtleusercontent.com/sfziyqlqaszf2a.png)


The decision tree in this class will be fully grown. Feel free to modify to prune.
