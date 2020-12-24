import numpy as np

# Data Type
uintType = np.uint8
floatType = np.float32

# Hopfield Class
class HOP(object):
    def __init__(self, N):
        # Bit Dimension
        self.N = N
        # Weight Matrix
        self.W = np.zeros((N, N), dtype = floatType)

    # Calculate Kronecker Square Product of [factor] itself OR use np.kron()
    def kroneckerSquareProduct(self, factor):
        ksProduct = np.zeros((self.N, self.N), dtype = floatType)

        # Calculate
        for i in range(0, self.N):
            ksProduct[i] = factor[i] * factor

        return ksProduct

    # Training a single stableState once a time, mainly to train [W]
    def trainOnce(self, inputArray):
        # Learn with normalization
        mean = float(inputArray.sum()) / inputArray.shape[0]
        self.W = self.W + self.kroneckerSquareProduct(inputArray - mean) / (self.N * self.N) / mean / (1 - mean)

        # Erase diagonal self-weight
        index = range(0, self.N)
        self.W[index, index] = 0.

    # Overall training function
    def hopTrain(self, stableStateList):
        # Preprocess List to Array type
        stableState = np.asarray(stableStateList, dtype = uintType)

        # Exception
        if np.amin(stableState) < 0 or np.amax(stableState) > 1:
            print('Vector Range ERROR!')
            return

        # Train
        if len(stableState.shape) == 1 and stableState.shape[0] == self.N:
            print('stableState count: 1')
            self.trainOnce(stableState)
        elif len(stableState.shape) == 2 and stableState.shape[1] == self.N:
            print('stableState count: ' + str(stableState.shape[0]))
            for i in range(0, stableState.shape[0]):
                self.trainOnce(stableState[i])
        else:
            print('SS Dimension ERROR! Training Aborted.')
            return
        print('Hopfield Training Complete.')

    # Run HOP to output
    def hopRun(self, inputList):
        # Preprocess List to Array type
        inputArray = np.asarray(inputList, dtype = floatType)

        # Exception
        if len(inputArray.shape) != 1 or inputArray.shape[0] != self.N:
            print('Input Dimension ERROR! Runing Aborted.')
            return

        # Run
        matrix = np.tile(inputArray, (self.N, 1))
        matrix = self.W * matrix
        ouputArray = matrix.sum(1)

        # Normalize
        m = float(np.amin(ouputArray))
        M = float(np.amax(ouputArray))
        ouputArray = (ouputArray - m) / (M - m)

        # Binary
        ''' \SWITCH/ : 1/-1 OR 1/0
        ouputArray[ouputArray < 0.5] = -1.
        ''' # \Division/
        ouputArray[ouputArray < 0.5] = 0.
        # ''' # \END/
        ouputArray[ouputArray > 0] = 1.

        return np.asarray(ouputArray, dtype = uintType)


# Utility routine for printing the input vector: [NperGroup] numbers each piece
def printFormat(vector, NperGroup):
    string = ''
    for index in range(len(vector)):
        if index % NperGroup == 0:
            ''' \SWITCH/ : Single-Row OR Multi-Row
            string += ' '
            ''' # \Division/
            string += '\n'
            # ''' # \END/

        # ''' \SWITCH/ : Image-Matrix OR Raw-String
        if str(vector[index]) == '0':
            string += ' '
        elif str(vector[index]) == '1':
            string += '*'
        else:
            string += str(vector[index])
        ''' # \Division/
        string += str(vector[index])
        # ''' # \END/
    string += '\n'
    print(string)

# DEMO of Hopfield Net
def HOP_demo():
    zero = [0, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1,
            0, 1, 1, 1, 0]
    one = [0, 1, 1, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 1, 0, 0,
           0, 0, 1, 0, 0]
    two = [1, 1, 1, 0, 0,
           0, 0, 0, 1, 0,
           0, 0, 0, 1, 0,
           0, 1, 1, 0, 0,
           1, 0, 0, 0, 0,
           1, 1, 1, 1, 1]

    a = [0, 0, 0, 1, 1, 1, 0, 0, 0,
         0, 0, 1, 1, 1, 1, 1, 0, 0,
         0, 1, 1, 1, 0, 1, 1, 1, 0,
         1, 1, 1, 0, 0, 0, 1, 1, 1,
         1, 1, 0, 0, 0, 0, 0, 1, 1,
         1, 1, 0, 0, 0, 0, 0, 1, 1,
         1, 1, 0, 0, 0, 0, 0, 1, 1,
         1, 1, 0, 0, 0, 0, 0, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 0, 0, 0, 0, 0, 1, 1,
         1, 1, 0, 0, 0, 0, 0, 1, 1,
         1, 1, 0, 0, 0, 0, 0, 1, 1,]
    c = [0, 0, 0, 1, 1, 1, 1, 1, 0,
         0, 0, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 1, 1, 1, 1, 1, 0,]
    l = [1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1,]
    #hop = HOP(5 * 6)
    #hop.hopTrain([zero, one, two])
    hop = HOP(9*12)
    hop.hopTrain([a, c, l])

    test_a = [0, 0, 0, 1, 1, 1, 0, 0, 0,
              0, 0, 1, 0, 0, 0, 1, 0, 0,
              0, 1, 0, 1, 0, 1, 1, 1, 0,
              1, 0, 1, 0, 0, 0, 1, 0, 1,
              1, 1, 0, 0, 0, 0, 0, 1, 1,
              0, 1, 0, 0, 0, 0, 0, 1, 0,
              1, 1, 0, 0, 0, 0, 0, 1, 0,
              0, 1, 0, 0, 0, 0, 0, 1, 1,
              1, 1, 0, 1, 1, 1, 1, 1, 0,
              1, 1, 0, 0, 0, 0, 0, 1, 1,
              1, 0, 0, 0, 0, 0, 0, 0, 1,
              1, 1, 0, 0, 0, 0, 0, 0, 1,]
    print('test_a:')
    printFormat(test_a, 9)
    result = hop.hopRun(test_a)
    print('Recovered:')
    printFormat(result, 9)

    '''
    half_zero = [0, 1, 1, 1, 0,
                 1, 0, 0, 0, 1,
                 1, 0, 0, 0, 1,
                 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0]
    print('Half-Zero:')
    printFormat(half_zero, 5)
    result = hop.hopRun(half_zero)
    print('Recovered:')
    printFormat(result, 5)

    half_two = [0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 1, 1, 0, 0,
                1, 0, 0, 0, 0,
                1, 1, 1, 1, 1]
    print('Half-Two:')
    printFormat(half_two, 5)
    result = hop.hopRun(half_two)
    print('Recovered:')
    printFormat(result, 5)

    half_two = [1, 1, 1, 0, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0]
    print('Another Half-Two:')
    printFormat(half_two, 5)
    result = hop.hopRun(half_two)
    print('Recovered:')
    printFormat(result, 5)
    '''

##########################
if __name__ == '__main__':
    HOP_demo()