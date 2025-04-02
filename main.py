# Barış Utku Ünsal - 2315604

# shine131.jpg image is removed from the dataset as it caused problems during implementation. Remove the file during
# testing otherwise the code will not compile.

import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


# Method to load the database and name the class labels according to file names.
def loadDatabase():
    directory = "Dataset"
    if not os.path.exists(directory):  # Checking the existence of directory if directory does not exist, exit program.
        print("The Dataset directory does not exist!")
        raise SystemExit(0)
    classLabels = []
    images = {}

    for file in os.listdir(directory):
        labelName = re.sub("[0-9]+\.[A-Za-z]+", "", file)  # Checking the file name to create label name using regex.
        if (labelName not in classLabels):
            classLabels.append(labelName)
            images[re.sub("[0-9]+\.[A-Za-z]+", "", file)] = []
        image = cv2.imread("Dataset/" + file)
        images[classLabels[classLabels.index(labelName)]].append(image)
    for i in range(0, len(classLabels)):
        images[classLabels[i]] = np.asarray(images[classLabels[i]], dtype=object)
    return images, classLabels


# Method to find the frequency of each bin in the image for hist_features.
def binFrequency(n, binAmount):
    totalAmount = 0
    frequencyArray = np.zeros([binAmount])
    for amount in n:  # Finding the total amount of values in the histogram for frequency calculation.
        totalAmount += amount
    i = 0
    for binValue in n:
        freq = binValue / totalAmount
        frequencyArray[i] = freq
        i += 1
    return frequencyArray


# Method to extract histogram features from images in the dataset.
def hist_features(imageList, binAmount):
    features = []
    for image in imageList:
        grayscaleImg = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Converting to grayscale.
        flattenedImg = grayscaleImg.flatten()  # Flattening the image to generate histogram from 1-D array.
        n, bins, patches = plt.hist(flattenedImg, bins=binAmount, range=(0, 256))
        binFrequencies = binFrequency(n, binAmount)
        features.append(binFrequencies)
    features = np.asarray(features,
                          dtype=object)  # Converting the list into numpy array as this data type is used in other methods.
    plt.close()
    return features


# Method to generate training, validation and training sets.
def formSets(features):
    trainingSet = {}
    validationSet = {}
    testingSet = {}

    labels = list(features.keys())
    for labelName in labels:
        trainingSet[labelName] = []
        validationSet[labelName] = []
        testingSet[labelName] = []
    # Taking 50% of each class for training, remaining 25% for validation and the last 25% for testing.
    for label in labels:
        for imageIndex in range(0, round(len(features[label]) / 2)):
            trainingSet[label].append(features[label][imageIndex])
        for imageIndex in range(round(len(features[label]) / 2),
                                round(len(features[label]) / 2) + round(len(features[label]) / 4)):
            validationSet[label].append(features[label][imageIndex])
        for imageIndex in range(round(len(features[label]) / 2) + round(len(features[label]) / 4),
                                len(features[label])):
            testingSet[label].append(features[label][imageIndex])

    return trainingSet, validationSet, testingSet


# Converting the information in sets to data types used in calculations. Had to create this method further down the
# development as a temporary patch, planned to integrate with formSets method. setType variable is used to detect
# which type of feature is being converted.
def createTrainingInfo(givenSet, setType):
    if setType == 0:
        data = []
        responses = []
        for classLabel in classLabels:
            for imageFeature in givenSet[classLabel]:
                for i in range(len(imageFeature)):
                    data.append([float(i + 1), imageFeature[i]])
                    responses.append(float(classLabels.index(classLabel)))
        responses = np.asarray(responses, dtype="float32")
        data = np.asarray(data, dtype="float32")
        return data, responses
    elif setType == 1:
        rData = []
        gData = []
        bData = []
        rResponses = []
        gResponses = []
        bResponses = []
        for classLabel in classLabels:
            for imageFeature in givenSet[classLabel]:
                for i in range(len(imageFeature)):
                    if (i < 10):
                        rData.append([float(i + 1), imageFeature[i]])
                        rResponses.append(float(classLabels.index(classLabel)))
                    elif (i < 20):
                        gData.append([float((i + 1) % 10), imageFeature[i]])
                        gResponses.append(float(classLabels.index(classLabel)))
                    elif (i < 30):
                        bData.append([float((i + 1) % 10), imageFeature[i]])
                        bResponses.append(float(classLabels.index(classLabel)))
        rResponses = np.asarray(rResponses, dtype="float32")
        gResponses = np.asarray(gResponses, dtype="float32")
        bResponses = np.asarray(bResponses, dtype="float32")
        rData = np.asarray(rData, dtype="float32")
        gData = np.asarray(gData, dtype="float32")
        bData = np.asarray(bData, dtype="float32")
        colorData = np.array([rData, gData, bData], dtype=object)
        colorResponses = np.array([rResponses, gResponses, bResponses], dtype=object)
        return colorData, colorResponses


# Method to convert 1-D features array into an array of image features.
def featureToImageClass(features, binAmount):
    # Splitting the array into arrays with the length of each image feature.
    images = np.split(features, (len(features) / binAmount))
    resultImgClasses = []
    for featureClasses in images:
        values, counts = np.unique(featureClasses, return_counts=True)
        resultImgClasses.append(values[counts.argmax()])
    resultImgClasses = np.asarray(resultImgClasses)
    return resultImgClasses


# Method to validate training data and pick the best K value. Automatically picks the highest K value but can be
# converted into user input mode by removing the commented lines and commenting the current return line.
def validation(validationData, validationResponses, knn, kValues):
    if (np.ndim(validationData) == 2):
        expectedResult = featureToImageClass(validationResponses, binAmount)
        knnResults = []
        for kVal in kValues:
            ret, results, neighbours, dist = knn.findNearest(validationData, kVal)
            knnResults.append(featureToImageClass(results, binAmount))

        knnAccuracies = []
        for result in knnResults:
            values, counts = np.unique(result == expectedResult, return_counts=True)
            knnAccuracies.append(float((counts[np.where(values == True)] / len(result)) * 100))

        plt.figure()
        plt.plot(kValues, knnAccuracies)
        plt.title("Accuracies of different k values")
        plt.show()

    elif (np.ndim(validationData) == 3):
        expectedResult = featureToImageClass(validationResponses.flatten(), binAmount)
        knnResults = []

        colorData = np.asarray(np.concatenate((validationData[0], validationData[1], validationData[2])),
                               dtype=np.float32)

        for kVal in kValues:
            ret, results, neighbours, dist = knn.findNearest(colorData, kVal)
            knnResults.append(featureToImageClass(results, binAmount))

        knnAccuracies = []
        for result in knnResults:
            knnAccuracies.append(np.mean(result.flatten() == expectedResult) * 100)

        plt.figure()
        plt.plot(kValues, knnAccuracies)
        plt.title("Accuracies of different k values in color histogram")
        plt.show()

    # userK = input("Choose the k value to be used in testing: " + str(kValues))
    # return userK

    return kValues[knnAccuracies.index(max(knnAccuracies))]


def training(trainData, trainResponses, validationData, validationResponses):
    # Detecting the feature type by checking the dimensions of the data array.
    if (np.ndim(trainData) == 2):

        # Generating arrays for each class to show to user using matplotlib.
        cloudy = trainData[trainResponses.ravel() == 0]
        shine = trainData[trainResponses.ravel() == 1]
        sunrise = trainData[trainResponses.ravel() == 2]

        plt.title("Training Set Visualization")
        plt.scatter(cloudy[:, 0], cloudy[:, 1], 10, 'r', '^')
        plt.scatter(shine[:, 0], shine[:, 1], 10, 'g', 'o')
        plt.scatter(sunrise[:, 0], sunrise[:, 1], 10, 'b', '*')
        plt.show()

        # Training the knn with training set to use in validation.
        knn = cv2.ml.KNearest_create()
        knn.train(trainData, cv2.ml.ROW_SAMPLE, trainResponses)
        kValues = [1, 3, 5, 7]

        return validation(validationData, validationResponses, knn, kValues)
    elif (np.ndim(trainData) == 3):

        # Generating arrays for each class to show to user using matplotlib.
        cloudy = np.array([trainData[0][trainResponses[0].ravel() == 0], trainData[1][trainResponses[0].ravel() == 0],
                           trainData[2][trainResponses[0].ravel() == 0]])
        shine = np.array([trainData[0][trainResponses[0].ravel() == 1], trainData[1][trainResponses[0].ravel() == 1],
                          trainData[2][trainResponses[0].ravel() == 1]])
        sunrise = np.array([trainData[0][trainResponses[0].ravel() == 2], trainData[1][trainResponses[0].ravel() == 2],
                            trainData[2][trainResponses[0].ravel() == 2]])

        # Each color channel is shown in a separate plot.
        for i in range(3):
            if i == 0:
                plt.title("Training Set Visualization - Red")
            elif i == 1:
                plt.title("Training Set Visualization - Green")
            elif i == 2:
                plt.title("Training Set Visualization - Blue")
            plt.scatter(cloudy[i][:, 0], cloudy[i][:, 1], 10, 'r', '^')
            plt.scatter(shine[i][:, 0], shine[i][:, 1], 10, 'g', 'o')
            plt.scatter(sunrise[i][:, 0], sunrise[i][:, 1], 10, 'b', '*')
            plt.show()
            plt.figure()

        # Concatenating the training and response data to create a 2-D and 1-D array to use in knn.
        colorData = np.asarray(np.concatenate((trainData[0], trainData[1], trainData[2])), dtype=np.float32)
        responses = np.asarray(np.concatenate((trainResponses[0], trainResponses[1], trainResponses[2])),
                               dtype=np.float32)

        knn = cv2.ml.KNearest_create()
        knn.train(colorData, cv2.ml.ROW_SAMPLE, responses)
        kValues = [1, 3, 5, 7]

        return validation(validationData, validationResponses, knn, kValues)


def testing(kVal, trainData, trainResponses, testData, testResponses):
    # Detecting the feature type by checking the dimensions of the data array.
    if (np.ndim(trainData) == 2):
        print("In testing for histogram features with k value:" + str(kVal))
        expectedResult = featureToImageClass(testResponses, binAmount)
        knn = cv2.ml.KNearest_create()
        knn.train(trainData, cv2.ml.ROW_SAMPLE, trainResponses)
        ret, results, neighbours, dist = knn.findNearest(testData, kVal)
        result = featureToImageClass(results, binAmount)

        # Calculating the accuracy by checking the amount of matches using np.unique and dividing it by the amount of
        # images. This part is done using np.mean in the color histogram part, did not update this to use that to show
        # my original implementation.
        values, counts = np.unique(result == expectedResult, return_counts=True)
        accuracy = float((counts[np.where(values == True)] / len(result)) * 100)



    elif (np.ndim(trainData) == 3):
        print("In testing for mystery features with k value:" + str(kVal))
        expectedResult = featureToImageClass(trainResponses.flatten(), binAmount)

        # Concatenating the training and response data to create a 2-D and 1-D array to use in knn.
        colorData = np.asarray(np.concatenate((trainData[0], trainData[1], trainData[2])), dtype=np.float32)
        responses = np.asarray(np.concatenate((trainResponses[0], trainResponses[1], trainResponses[2])),
                               dtype=np.float32)

        knn = cv2.ml.KNearest_create()
        knn.train(colorData, cv2.ml.ROW_SAMPLE, responses)
        ret, results, neighbours, dist = knn.findNearest(colorData, kVal)
        result = featureToImageClass(results, binAmount)

        # Using numpy mean method to calculate the accuracy.
        accuracy = np.mean(result.flatten() == expectedResult) * 100

    print("Accuracy is : " + str(accuracy))


# Write a comment in your code and answer these questions: Why did you choose this type of
# feature? How does it help to get good classification accuracy?
# I've chosen color histogram for my mystery feature as the given dataset has classes with significantly different
# colors. By checking the color values of any new images, it increases the accuracy compared to just grayscale histogram
# features as we have more values to compare with.
def mystery_features(imageList, binAmount):
    features = []
    for image in imageList:
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        reds = rgbImage[:, :, 0]
        greens = rgbImage[:, :, 1]
        blues = rgbImage[:, :, 2]
        nRed, bins, patches = plt.hist(reds.ravel(), bins=binAmount, range=(0, 256))
        nBlue, bins, patches = plt.hist(greens.ravel(), bins=binAmount, range=(0, 256))
        nGreen, bins, patches = plt.hist(blues.ravel(), bins=binAmount, range=(0, 256))
        features.append(np.concatenate((nRed, nBlue, nGreen)))
    features = np.asarray(features, dtype=object)
    plt.close()
    return features


if __name__ == '__main__':
    # Bin amount can be changed by changing the value the variable is assigned to. I've chosen 10 as it was the value
    # with the highest accuracy and fast processing time for hist_features.
    binAmount = 10
    images, classLabels = loadDatabase()

    # Generating empty dictionaries to insert the features into.
    histogramFeatures = {}
    mysteryFeatures = {}
    for labelName in classLabels:
        histogramFeatures[labelName] = hist_features(images[labelName], binAmount)
        mysteryFeatures[labelName] = mystery_features(images[labelName], binAmount)

    # Generating sets for hist_features and mystery_features
    trainingHistSet, validationHistSet, testingHistSet = formSets(histogramFeatures)
    trainingColorSet, validationColorSet, testingColorSet = formSets(mysteryFeatures)

    # Converting the sets into information to be used in the methods. As stated in this method's comment, this method
    # was implemented later during development in order to fix a bug I've encountered with the way I've implemented
    # sets via dictionaries. I planned to integrate it into the formSets method later but ran out of time.
    trainHistData, trainHistResponses = createTrainingInfo(trainingHistSet, 0)
    validationHistData, validationHistResponses = createTrainingInfo(validationHistSet, 0)
    testHistData, testHistResponses = createTrainingInfo(testingHistSet, 0)

    trainColorData, trainColorResponses = createTrainingInfo(trainingColorSet, 1)
    validationColorData, validationColorResponses = createTrainingInfo(validationColorSet, 1)
    testColorData, testColorResponses = createTrainingInfo(testingColorSet, 1)

    # Running training and testing
    kValueHist = training(trainHistData, trainHistResponses, validationHistData, validationHistResponses)
    testing(kValueHist, trainHistData, trainHistResponses, testHistData, testHistResponses)

    kValueColor = training(trainColorData, trainColorResponses, validationColorData, validationColorResponses)
    testing(kValueColor, trainColorData, trainColorResponses, testColorData, testColorResponses)
