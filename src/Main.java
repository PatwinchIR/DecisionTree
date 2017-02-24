import java.io.IOException;
import java.util.ArrayList;

public class Main {

    public static void validation(DecisionTree dt, int foldNum) {
        int trainNum = (dt.trainData.entries.size() / foldNum) * (foldNum - 1);
        int testNum = dt.trainData.entries.size() - trainNum;

        ArrayList<Double> accuracy = new ArrayList<>();

        for (int i = 0; i < foldNum; i ++) {
            DecisionTree fold = new DecisionTree();

            ArrayList<Integer> testIndex = new ArrayList<>();

            for (int p = 0; p < testNum; p ++) {
                Integer index = (int) (Math.random() * (dt.trainData.entries.size()));

                while (testIndex.contains(index)) {
                    index = (int) (Math.random() * (dt.trainData.entries.size()));
                }
                testIndex.add(index);

                fold.testData.entries.add(dt.trainData.entries.get(index));
            }
            for (int q = 0; q < dt.trainData.entries.size(); q ++) {
                if (testIndex.contains(new Integer(q))) {
                    continue;
                }
                fold.trainData.entries.add(dt.trainData.entries.get(q));
            }
            System.out.println("\nFold " + i + ": ");
            fold.startTrain();
            accuracy.add(fold.startTest());
        }

        double sum = 0;
        for (int i = 0; i < accuracy.size(); i ++) {
            sum += accuracy.get(i);
        }
        System.out.println("\nMean accuracy is: " + sum / accuracy.size());
    }

    public static void main(String[] args) throws IOException {
        DecisionTree dt = new DecisionTree();
        dt.loadData(true, "train.csv");
        dt.startTrain();
        dt.preorderTraversePrint(dt.start, dt.root, -1, false, true);
        dt.loadData(false, "test.csv");
        dt.startTest();
        dt.confusionMatrixPrint();
        System.out.println("\n\nStart 10 folds validation:");
        validation(dt, 10);
    }
}
