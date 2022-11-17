import java.util.Random;
import java.text.DecimalFormat;

class BackpropagationExample {

    //학습방법 설정
    private static final int INPUT_NEURONS = 4;
    private static final int HIDDEN_NEURONS1 = 20; //HIDDEN_NEURONS -> HIDDEN_NEURONS1, HIDDEN_NEURONS2
    private static final int HIDDEN_NEURONS2 = 14;
    private static final int OUTPUT_NEURONS = 14;

    private static final double LEARN_RATE = 0.2;    // Rho.
    private static final double NOISE_FACTOR = 0.45;
    private static final int TRAINING_REPS = 7000;

    //경로별 Weights 설정
    // Input to Hidden Weights (with Biases).
    private static double wih[][] = new double[INPUT_NEURONS + 1][HIDDEN_NEURONS1];

    // Hidden to Hidden Weights (with Biases).
    private static double whh[][] = new double[HIDDEN_NEURONS1 + 1][HIDDEN_NEURONS2];

    // Hidden to Output Weights (with Biases).
    private static double who[][] = new double[HIDDEN_NEURONS2 + 1][OUTPUT_NEURONS];

    // Activations.
    private static double inputs[] = new double[INPUT_NEURONS];
    private static double hidden1[] = new double[HIDDEN_NEURONS1]; //hidden -> hidden1, hidden2
    private static double hidden2[] = new double[HIDDEN_NEURONS2];
    private static double target[] = new double[OUTPUT_NEURONS];
    private static double actual[] = new double[OUTPUT_NEURONS];

    // Unit errors.
    private static double erro[] = new double[OUTPUT_NEURONS];
    private static double errh1[] = new double[HIDDEN_NEURONS1]; //errh -> errh1, errh2
    private static double errh2[] = new double[HIDDEN_NEURONS2];


    private static final int MAX_SAMPLES = 14;

    private static int trainInputs[][] = new int[][] {{1, 1, 1, 0},
            {1, 1, 0, 0},
            {0, 1, 1, 0},
            {1, 0, 1, 0},
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {1, 1, 1, 1},
            {1, 1, 0, 1},
            {0, 1, 1, 1},
            {1, 0, 1, 1},
            {1, 0, 0, 1},
            {0, 1, 0, 1},
            {0, 0, 1, 1}};

    private static int trainOutput[][] = new int[][]
            {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

    private static void NeuralNetwork()
    {
        int sample = 0;

        assignRandomWeights();

        // Train the network.
        for(int epoch = 0; epoch < TRAINING_REPS; epoch++) //학습 횟수만큼 반복
        {
            sample += 1;
            if(sample == MAX_SAMPLES){
                sample = 0;
            }

            for(int i = 0; i < INPUT_NEURONS; i++)
            {
                inputs[i] = trainInputs[sample][i];
            } // i

            for(int i = 0; i < OUTPUT_NEURONS; i++)
            {
                target[i] = trainOutput[sample][i];
            } // i

            feedForward();

            backPropagate();

        } // epoch

        getTrainingStats();

        System.out.println("\nTest network against original input:");
        testNetworkTraining();

        System.out.println("\nTest network against noisy input:");
        testNetworkWithNoise1();

        return;
    }

    private static void getTrainingStats()
    {
        double sum = 0.0;
        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                inputs[j] = trainInputs[i][j];
            } // j

            for(int j = 0; j < OUTPUT_NEURONS; j++)
            {
                target[j] = trainOutput[i][j];
            } // j

            feedForward();

            if(maximum(actual) == maximum(target)){
                sum += 1;
            }else{
                System.out.println(inputs[0] + "\t" + inputs[1] + "\t" + inputs[2] + "\t" + inputs[3]);
                System.out.println(maximum(actual) + "\t" + maximum(target));
            }
        } // i

        System.out.println("Network is " + ((double)sum / (double)MAX_SAMPLES * 100.0) + "% correct.");

        return;
    }

    private static void testNetworkTraining()
    {
        // This function simply tests the training vectors against network.
        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                inputs[j] = trainInputs[i][j];
            } // j

            feedForward();

            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                System.out.print(inputs[j] + "\t");
            } // j

            System.out.print("Output: " + maximum(actual) + "\n");
        } // i

        return;
    }

    private static void testNetworkWithNoise1()
    {
        // This function adds a random fractional value to all the training
        // inputs greater than zero.
        DecimalFormat dfm = new java.text.DecimalFormat("###0.0");

        for(int i = 0; i < MAX_SAMPLES; i++)
        {
            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                inputs[j] = trainInputs[i][j] + (new Random().nextDouble() * NOISE_FACTOR);
            } // j

            feedForward();

            for(int j = 0; j < INPUT_NEURONS; j++)
            {
                System.out.print(dfm.format(((inputs[j] * 1000.0) / 1000.0)) + "\t");
            } // j
            System.out.print("Output: " + maximum(actual) + "\n");
        } // i

        return;
    }

    private static int maximum(final double[] vector)
    {
        // This function returns the index of the maximum of vector().
        int sel = 0;
        double max = vector[sel];

        for(int index = 0; index < OUTPUT_NEURONS; index++)
        {
            if(vector[index] > max){
                max = vector[index];
                sel = index;
            }
        }
        return sel;
    }

    private static void feedForward() //정방향 학습
    {
        double sum = 0.0;

        // Calculate input to hidden layer.
        for(int hid = 0; hid < HIDDEN_NEURONS1; hid++)
        {
            sum = 0.0;
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                sum += inputs[inp] * wih[inp][hid];
            } // inp

            sum += wih[INPUT_NEURONS][hid]; // Add in bias.
            hidden1[hid] = sigmoid(sum);
        } // hid


        // Calculate hidden to hidden layer.
        for(int hid2 = 0; hid2 < HIDDEN_NEURONS2; hid2++)
        {
            sum = 0.0;
            for(int hid1 = 0; hid1 < HIDDEN_NEURONS1; hid1++)
            {
                sum += hidden1[hid1] * whh[hid1][hid2];
            } // hid1

            sum += whh[HIDDEN_NEURONS1][hid2]; // Add in bias.
            hidden2[hid2] = sigmoid(sum);
        } // hid2
        

        // Calculate the hidden to output layer.
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            sum = 0.0;
            for(int hid = 0; hid < HIDDEN_NEURONS2; hid++)
            {
                sum += hidden2[hid] * who[hid][out];
            } // hid

            sum += who[HIDDEN_NEURONS2][out]; // Add in bias.
            actual[out] = sigmoid(sum);
        } // out
        return;
    }

    private static void backPropagate() //역방향 학습
    {
        // Calculate the output layer error (step 3 for output cell).
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            erro[out] = (target[out] - actual[out]) * sigmoidDerivative(actual[out]);
        }

        // Calculate the hidden layer1 error (step 3 for hidden cell).
        for(int hid1 = 0; hid1 < HIDDEN_NEURONS1; hid1++)
        {
            errh1[hid1] = 0.0;
            for(int hid2 = 0; hid2 < HIDDEN_NEURONS2; hid2++)
            {
                errh1[hid1] += erro[hid2] * whh[hid1][hid2];
            }
            errh1[hid1] *= sigmoidDerivative(hidden1[hid1]);
        }

        // Calculate the hidden layer2 error (step 3 for hidden cell).
        for(int hid = 0; hid < HIDDEN_NEURONS2; hid++)
        {
            errh2[hid] = 0.0;
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                errh2[hid] += erro[out] * who[hid][out];
            }
            errh2[hid] *= sigmoidDerivative(hidden2[hid]);
        }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Update the weights for the output layer (step 4). out->hid2
        for(int out = 0; out < OUTPUT_NEURONS; out++)
        {
            for(int hid = 0; hid < HIDDEN_NEURONS2; hid++)
            {
                who[hid][out] += (LEARN_RATE * erro[out] * hidden2[hid]);
            } // hid2
            who[HIDDEN_NEURONS2][out] += (LEARN_RATE * erro[out]); // Update the bias.
        } // out

        // Update the weights for the output layer (step 4). hid2->hid1
        for(int hid2 = 0; hid2 < HIDDEN_NEURONS2; hid2++)
        {
            for(int hid1 = 0; hid1 < HIDDEN_NEURONS1; hid1++)
            {
                whh[hid1][hid2] += (LEARN_RATE * errh2[hid2] * hidden1[hid1]);
            } // hid1
            whh[HIDDEN_NEURONS1][hid2] += (LEARN_RATE * errh2[hid2]); // Update the bias.
        } // hid2

        // Update the weights for the hidden layer (step 4). hid1->inp
        for(int hid = 0; hid < HIDDEN_NEURONS1; hid++)
        {
            for(int inp = 0; inp < INPUT_NEURONS; inp++)
            {
                wih[inp][hid] += (LEARN_RATE * errh1[hid] * inputs[inp]);
            } // inp
            wih[INPUT_NEURONS][hid] += (LEARN_RATE * errh1[hid]); // Update the bias.
        } // hid1
        return;
    }

    private static void assignRandomWeights()
    {
        for(int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here. inp->hid1
        {
            for(int hid = 0; hid < HIDDEN_NEURONS1; hid++)
            {
                // Assign a random weight value between -0.5 and 0.5
                wih[inp][hid] = new Random().nextDouble() - 0.5;
            } // hid
        } // inp

        for(int hid1 = 0; hid1 <= HIDDEN_NEURONS1; hid1++) // Do not subtract 1 here. hid1->hid2
        {
            for(int hid2 = 0; hid2 < HIDDEN_NEURONS2; hid2++)
            {
                // Assign a random weight value between -0.5 and 0.5
                whh[hid1][hid2] = new Random().nextDouble() - 0.5;
            } // hid2
        } // hid1

        for(int hid = 0; hid <= HIDDEN_NEURONS2; hid++) // Do not subtract 1 here. hid2->out
        {
            for(int out = 0; out < OUTPUT_NEURONS; out++)
            {
                // Assign a random weight value between -0.5 and 0.5
                who[hid][out] = new Random().nextDouble() - 0.5;
            } // out
        } // hid
        return;
    }

    private static double sigmoid(final double val)
    {
        return (1.0 / (1.0 + Math.exp(-val)));
    }

    private static double sigmoidDerivative(final double val)
    {
        return (val * (1.0 - val));
    }

    public static void main(String args[]) {
        NeuralNetwork();
    }
}
