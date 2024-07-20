from core.Extra import readConfiguration
from core.PrepareData import createBulletin
from core.Model import prepareTrainTest, runEstimator, makePrediction
from core.Visualize import (
    plotDataStatistics,
    plotValidationCurve,
    plotLearningCurve,
    plotPermutationImportance,
    plotMisfits,
    plotPredictionVsTest,
    plotMAE
)
from core.Predict import predict


class Main:
    def __init__(self):
        self.config = readConfiguration()

    def prepareInputs(self):
        createBulletin(self.config)

    def runModel(self):
        prepareTrainTest(self.config)
        runEstimator(self.config)
        makePrediction(self.config)

    def visualize(self):
        # To visulize the prediction performance of the model
        plotPredictionVsTest(self.config)
        # To visulize the amount of data for each feature
        # plotDataStatistics(self.config)
        # To visulize the effect of each hyperparameter in Under/Over fitting
        # plotValidationCurve(self.config)
        # To visulize the effect of adding more data in Under/Over fitting
        # plotLearningCurve(self.config)
        # To visulize the importance of each feature in estimated model
        # plotPermutationImportance()
        # To visulize the differnces between real and predictions
        plotMisfits(self.config)
        plotMAE(self.config)

    def locate(self):
        predict(self.config)


if "__main__" == __name__:
    app = Main()
    app.prepareInputs()
    app.runModel()
    app.visualize()
    app.locate()
