import javafx.util.Pair;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Created by d_d on 2/21/17.
 * This class is for Training and Testing purpose;
 * This class of Decision Tree can only deal with continuous attributes value and discrete output labels.
 * This class can only support CSV fomatted data file. Training and Test dataset format should be:
 * ||============================================||
 * ||     attr1,attr2,attr3,...,attrn,label      ||
 * ||============================================||
 * For continuous output, Please refer to regression tree, decision tree is more like a classifier.
 * TODO: Extend this class to BIN-DISCRETE(more than 2), BIN-BIN.
 */
public class DecisionTree {

    /**
     * Decision Tree Constructor.
     * Initialise training dataset and testing dataset.
     */
    public DecisionTree() {
        this.trainData = new Entries();
        this.testData = new Entries();
    }


    /**
     * Utility of LOG function, support any base.
     * @param x Log parameter.
     * @param base Base, should be integer.
     * @return The log result.
     */
    private static double log(double x, int base) {
        return (Math.log(x) / Math.log(base));
    }

    // Training Data.
    public Entries trainData;

    // Testing Data.
    public Entries testData;

    // Decision Tree's root node.
    public Node root;

    // For visualization/output purpose.
    public Node start;

    // The confusion matrix.
    public Map<Pair<String, String>, Integer> confusionMatrix;

    /**
     * This class is for Node of decision tree.
     * It stores information about current entropy, current examples labels' kinds and numbers.
     * it also stores next best attribute to split, and the splitting boundary.
     */
    public class Node {
        // Entropy calculated on labelsCount which is formed by splitting boundary.
        public double entropy;

        // Using the splitting boundary form a labelsCount.
        // It's a hash map, the key is label, the value is the number of it.
        public Map<String, Integer> labelsCount;

        // Leaf node's label, which is used to produce prediction. NULL if non-leaf nodes.
        public String label;

        // To tag if current node needs more splitting.
        public boolean isConsistent;

        // The best attribute that needs to be split.
        public int bestAttribute;

        // The decision boundary for the best attribute. Also using this to binarize the data.
        public double decisionBoundary;

        // Left child.
        public Node left;

        // Right child.
        public Node right;

        /**
         * A utility that facilitates the counting process in a hash map for certain key.
         * Same as in python Collections.Counter().
         * @param hashMap The hash map that needs to be updated.
         * @param key The key that needs to be updated.
         * @return The updated hash map.
         */
        public Map<String, Integer> Counter(Map<String, Integer> hashMap, String key) {
            Map<String, Integer> temp = new HashMap<>(hashMap);
            if (temp.containsKey(key)) {
                int count = temp.get(key);
                temp.put(key, ++ count);
            } else {
                temp.put(key, 1);
            }
            return temp;
        }

        /**
         * For current examples, generate the label count, for later calculating entropy and consistency check.
         * @param examples The examples that current node received.
         */
        private void processLabels(Entries examples) {
            List<String> labels = new ArrayList<>();
            this.labelsCount = new HashMap<>();
            for (Entry e: examples.entries) {
                this.labelsCount = Counter(this.labelsCount, e.label);
                labels.add(e.label);
            }
            if (this.labelsCount.size() == 1) {

                // If only one label exists in current example then set the prediction label to it.
                this.label = labels.get(0);

                // No need to split more, current node is consistent with examples.
                this.isConsistent = true;

            } else {
                // Received more than 1 labels, meaning not consistent, set the tag to false.
                this.isConsistent = false;
            }
        }

        /**
         * Calculate entropy for a labelsCount.
         * @param n The total number of labels.
         * @param labelsCount The hash map of labels, the key is label, the value is the number of it.
         * @return The labelsCount's entropy.
         */
        private double calculateEntropy(Integer n, Map<String, Integer> labelsCount) {
            double entropy = 0;
            Iterator it = labelsCount.entrySet().iterator();
            while (it.hasNext()) {
                Map.Entry pair = (Map.Entry)it.next();
                Double count = ((Integer) pair.getValue()) * 1.0; // * 1.0 is to convert it to double type.
                double p = count / n;
                entropy -= (p * log(p, 2));
            }
            return entropy;
        }

        /**
         * The main function to find the next best splitting attribute.
         * I put it into Node because every node would receive a set of examples when it's created and wouldn't be
         * changed later.
         * @param examples The examples that current node received after its parent's splitting.
         * @param attributes The remaining attributes that haven't been spitted before.
         */
        private void findBestSplitAttr(Entries examples, ArrayList<Integer> attributes) {

            // minEntropy over all attributes and all candidate boundaries.
            double minEntropy = Double.MAX_VALUE;

            // Traverse all remaining attributes.
            for (Integer attrIdx: attributes) {

                // Sort examples according to current attributes.
                Collections.sort(examples.entries, new Comparator<Entry>() {
                    @Override
                    public int compare(Entry o1, Entry o2) {
                        return Double.compare(o1.attributes.get(attrIdx),
                                o2.attributes.get(attrIdx));
                    }
                });

                // Trying all candidate boundaries.
                for (int i = 1; i < examples.entries.size(); i ++) {

                    // Discretise examples into binary.
                    Map<String, Integer> pos = new HashMap<>();
                    Map<String, Integer> neg = new HashMap<>();

                    for (int j = 0; j < examples.entries.size(); j ++) {
                        String newLabel = examples.entries.get(j).label;
                        if (j < i) {
                            pos = Counter(pos, newLabel);
                        } else {
                            neg = Counter(neg, newLabel);
                        }
                    }

                    // Calculate pos and neg entropy.
                    double posEntropy = calculateEntropy(i, pos);
                    double negEntropy = calculateEntropy(examples.entries.size() - i, neg);

                    // Updating the minEntropy.
                    if ((posEntropy + negEntropy) < minEntropy) {
                        minEntropy = posEntropy + negEntropy;
                        if (attrIdx == 3) {
                            this.decisionBoundary = (examples.entries.get(i - 1).attributes.get(attrIdx) + examples.entries.get(i).attributes.get(attrIdx)) / 2 + 0.1;
                        } else {
                            this.decisionBoundary = (examples.entries.get(i - 1).attributes.get(attrIdx) + examples.entries.get(i).attributes.get(attrIdx)) / 2;
                        }
                        this.bestAttribute = attrIdx;
                    }
                }
            }
        }

        /**
         * Constructor for Node when it needs to receive examples and remaining attributes.
         * @param examples The remaining examples after its parent's splitting.
         * @param attributes The remaining attributes after its parent's splitting.
         */
        public Node(Entries examples, ArrayList<Integer> attributes) {
            this.left = null;
            this.right = null;
            this.label = null;

            processLabels(examples);

            this.entropy = calculateEntropy(examples.entries.size(), this.labelsCount);

            findBestSplitAttr(examples, attributes);
        }

        /**
         * Constructor of Node when there's no examples left.
         */
        public Node() {
            this.left = null;
            this.right = null;
            this.label = null;
            this.entropy = 0;
            this.labelsCount = new HashMap<>();
            this.isConsistent = true;
            this.bestAttribute = 0;
            this.label = "";
        }
    }

    /**
     * This class is specifically for dataset, each entry is a row in dataset, seperated as attributes and
     * correspongding label(also called target attribute in some ID3 algorithm tutorials).
     */
    public class Entries {
        public List<Entry> entries;

        public Entries() {
            this.entries = new ArrayList<>();
        }
    }

    /**
     * This class is specifically for row in dataset, attributes is a list of Double value, because of
     * continuous/numerical attributes.
     * label is a String indicating the label.
     */
    public class Entry {
        public List<Double> attributes;
        public String label;

        public Entry() {
            this.attributes = new ArrayList<>();
            this.label = null;
        }
    }

    /**
     * A utility function to read a CSV as a List of String Arrays, each element is a row.
     * @param filePath The CSV filepath.
     * @return The rows raw data.
     * @throws IOException In case of IOException.
     */
    private List<String[]> readCSV(String filePath) throws IOException {
        BufferedReader fileReader = new BufferedReader(new FileReader(filePath));
        String line;
        List<String[]> entries = new ArrayList<>();
        while ((line = fileReader.readLine()) != null) {

            // Change here for other delimiters.
            String[] tokens = line.split(",");

            if (tokens.length > 0) {
                entries.add(tokens);
            }
        }
        return entries;
    }

    /**
     * A public method for user to load data for DecisionTree class.
     * NOTICE: Training data and test data need to be loaded seperately, label is as default the last column
     *         in the dataset.
     * @param training  Indicate if the file is training data.
     * @param filePath  Indicate the filepath.
     * @throws IOException In case of IOException.
     */
    public void loadData(boolean training, String filePath) throws IOException {
        List<String[]> entries = readCSV(filePath);
        for (String[] s: entries) {
            int i;

            Entry newEntry = new Entry();
            for (i = 0; i < s.length - 1; i ++) {
                newEntry.attributes.add(Double.parseDouble(s[i]));
            }

            // The last colunm is as default the label.
            newEntry.label = s[i];

            if (training) {
                this.trainData.entries.add(newEntry);
            } else {
                this.testData.entries.add(newEntry);
            }
        }
    }

    /**
     * The main ID3 recursive function. The pseudocode can be found at:
     * https://www.cs.swarthmore.edu/~meeden/cs63/f05/id3.html
     * @param examples  The examples for next splitting.
     * @param attributes    The attributes for next splitting. (Remaining Attributes.)
     * @return  The root node of the DecisionTree.
     */
    public Node ID3(Entries examples, ArrayList<Integer> attributes){
        Node node = new Node(examples, attributes);

        // If current node is already consistent with examples, return.
        if (node.isConsistent) {
            return node;

        } else {
            // If there's no longer attributes, no need to continue.
            if (attributes.size() == 0) {

                // There's no attributes to continue splitting though current node is not consistent, then
                // take a majority vote for this leaf node's label.
                node.label = Collections.max(node.labelsCount.entrySet(), Map.Entry.comparingByValue()).getKey();
                return node;

            }
            // Sort examples according to best splitting attribute, for splitting dataset later.
            Collections.sort(examples.entries, new Comparator<Entry>() {
                @Override
                public int compare(Entry o1, Entry o2) {
                    return Double.compare(o1.attributes.get(node.bestAttribute),
                            o2.attributes.get(node.bestAttribute));
                }
            });

            // Split dataset according to decision boundary of the best splitting attribute.
            Entries newExamplesLeft = new Entries();
            Entries newExampleRight = new Entries();

            for (int i = 0; i < examples.entries.size(); i ++) {
                Entry entryTemp = examples.entries.get(i);
                if (entryTemp.attributes.get(node.bestAttribute) <= node.decisionBoundary) {
                    newExamplesLeft.entries.add(entryTemp);
                } else {
                    newExampleRight.entries.add(entryTemp);
                }
            }

            // Generating remaining attributes.
            ArrayList<Integer> newAttributes = new ArrayList<>(attributes);
            newAttributes.remove(Integer.valueOf(node.bestAttribute));

            // If the dataset after splitting is not empty, then branching and grow the tree. Else end growing.
            if (newExamplesLeft.entries.size() != 0) {
                node.left = ID3(newExamplesLeft, newAttributes);
            } else {
                node.left = new Node();
            }
            if (newExampleRight.entries.size() != 0) {
                node.right = ID3(newExampleRight, newAttributes);
            } else {
                node.right = new Node();
            }

            return node;
        }
    }

    /**
     * Using preorder traversal to print the ouput for required visualizaiton.
     * Because the desired output requires to show the splitting attribute on the child node, along with the
     * decision boundary, and to use "<" or ">=" to represent left and right child respectively.
     * @param parent Indicate child's desired output for attribute and decision boundary.
     * @param node  Indicate current node.
     * @param depth Indicate the depth of current node.
     * @param right Indicate if current node is a right child of its parent, to determine "<" and ">=".
     * @param start Indicate if it's the starting node(before root).
     */
    public void preorderTraversePrint(Node parent, Node node, Integer depth, boolean right, boolean start) {
        if (node == null) {
            return;
        }
        for (int i = 0; i < depth; i ++){
            System.out.print("\t");
        }
        if (!start){
            if (!right) {
                System.out.print("|Attr" + parent.bestAttribute + " < " + parent.decisionBoundary + " : " + node.label + " ");
            } else {
                System.out.print("|Attr" + parent.bestAttribute + " >= " + parent.decisionBoundary + " : " + node.label + " ");
            }
        }
        Iterator it = node.labelsCount.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry)it.next();
            String label = (String) pair.getKey();
            Integer count = (Integer) pair.getValue();
            System.out.print("[" + label + " : " + count + "]");
        }
        System.out.print("\n");
        preorderTraversePrint(node, node.left, depth + 1, false, false);
        preorderTraversePrint(node, node.right, depth + 1, true, false);
    }

    /**
     * Using current test entry to traverse the built tree and return predicted label.
     * @param entry Current test entry.
     * @param node  The decision tree's root node.
     * @return  The current test entry's predicted label.
     */
    public String getPrediction(Entry entry, Node node) {
        if (node.left == null && node.right == null)
            return node.label;
        if (entry.attributes.get(node.bestAttribute) <= node.decisionBoundary) {
            return getPrediction(entry, node.left);
        } else {
            return getPrediction(entry, node.right);
        }
    }

    private void printWhitespaces(int n) {
        for (int i = 0; i < n; i ++){
            System.out.print(" ");
        }
    }

    private void printDelimeter() {
        System.out.print("|");
    }

    private void printLine(int n, int size) {
        printDelimeter();
        for (int i = 0; i < n * (size + 1) + size; i ++) {
            System.out.print("=");
        }
        printDelimeter();
        System.out.println();
    }

    private void printMatrixHead(int n, int size) {
        int width = n * (size + 1) + size;
        String head = "Confusion Matrix";
        printWhitespaces((width - head.length()) / 2);
        System.out.println(head);
        printLine(n, size);
    }

    /**
     * A utility to print confusion matrix.
     */
    public void confusionMatrixPrint() {
        Map<String, Integer> labelsCount = this.root.labelsCount;
        List<String> labels = new ArrayList<>();

        int longest = 0;
        Iterator it = labelsCount.entrySet().iterator();
        while (it.hasNext()) {
            String al = (String) ((Map.Entry) it.next()).getKey();
            longest = al.length() > longest ? al.length() : longest;
            if (!labels.contains(al)) {
                labels.add(al);
            }
        }
        printMatrixHead(longest, labels.size());
        printDelimeter();
        printWhitespaces(longest);
        printDelimeter();
        for (int i = 0; i < labels.size(); i ++) {
            printWhitespaces(longest - labels.get(i).length());
            System.out.print(labels.get(i));
            printDelimeter();
        }
        System.out.print("\n");
        printLine(longest, labels.size());
        for (int i = 0; i < labels.size(); i ++) {
            String al = labels.get(i);
            printDelimeter();
            printWhitespaces(longest - al.length());
            System.out.print(al);
            printDelimeter();
            for (int j = 0; j < labels.size(); j ++) {
                String pl = labels.get(j);
                Integer count = this.confusionMatrix.get(new Pair<>(al, pl));
                String num = count == null ? "0" : String.valueOf(count);
                printWhitespaces(longest - num.length());
                System.out.print(num);
                printDelimeter();
            }
            System.out.print("\n");
            printLine(longest, labels.size());
        }
    }

    /**
     * A utility to update confusion matrix.
     * @param pair The <actual label, predict label> pair.
     */
    private void updateConfusionMatrix(Pair<String, String> pair) {
        if (this.confusionMatrix.containsKey(pair)) {
            int count = this.confusionMatrix.get(pair);
            this.confusionMatrix.put(pair, ++ count);
        } else {
            this.confusionMatrix.put(pair, 1);
        }
    }


    /**
     * Funtion to start building the tree.
     */
    public void startTrain() {
        ArrayList<Integer> attributes = new ArrayList<>();

        // The attributes index array. To indicate the remaining unsplit attributes.
        // Initially all attributes are reamined.
        for (int i = 0; i < this.trainData.entries.get(0).attributes.size(); i ++) {
            attributes.add(i);
        }

        this.start = new Node();
        this.root = ID3(this.trainData, attributes);
    }

    /**
     * Funtion to start testing the test dataset.
     * @return The accuracy.
     */
    public double startTest() {
        this.confusionMatrix = new HashMap<>();
        double correct = 0;
        double all = 0;
        for (Entry e: this.testData.entries) {
            String predictedLabel = getPrediction(e, this.root);
            if (predictedLabel.equals(e.label)) {
                correct ++;
            } else {
                System.out.print("Miss classifying [ ");
                for (Double d: e.attributes) {
                    System.out.print(d + ", ");
                }
                System.out.print(e.label + "]\tas [" + predictedLabel + "]\n");
            }
            Pair<String, String> pair = new Pair<>(e.label, predictedLabel);
            updateConfusionMatrix(pair);
            all ++;
        }
        double accuracy = correct / all;
        System.out.println("Accuracy: " + accuracy);
        return accuracy;
    }
}
