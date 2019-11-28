import trainer
import unittest
import pandas

class TrainerTest(unittest.TestCase):
    def setUp(self):
        self.trainfile = '../../data/trainingandtestdata/data_train.csv'
        self.testfile = '../../data/trainingandtestdata/data_test.csv'
        self.testing_data_train = '../../data/trainingandtestdata/testing/testing_data_train.csv'
        self.key_pos = 0
        self.text_pos = 5
        self.file_size = 100000
        self.training_sample_size = 100
        print()

    def test_load_data(self):
        # Act and assert no errors upon loading
        trainer.load_data(self.training_sample_size, self.file_size, self.trainfile, self.testfile)

    def test_corpusGenerator(self):
        # Arrange
        data_test = pandas.read_csv(self.testfile, header=None, usecols=[self.key_pos,self.text_pos], skiprows=list(range(5,500)))
        corpus_correct = ['stellargirl loooooooovvvvvvee kindle2 dx cool 2 fantast right', 'read kindle2 love lee child good read', 'ok first asses kindle2 fuck rock', 'kenburbari love kindle2 mine month never look back new big one huge need remors', 'mikefish fair enough kindle2 think perfect']
        # Act
        corpus_test = trainer.corpusGenerator(data_test)
        # Assert
        self.assertEqual(corpus_test, corpus_correct)

    def test_train_cv(self):
        # Arrange
        cv_act_feature_names = ['10', '30am', '4am', 'angryfeet', 'anywher', 'attack', 'away', 'back', 'blame', 'break', 'btw', 'bud', 'cant', 'doctor', 'drive', 'edinburgh', 'elliott', 'em91', 'ex', 'fed', 'friday', 'fring', 'get', 'girl', 'go', 'goe', 'good', 'hahahahaha', 'home', 'hope', 'hospit', 'howliet', 'hungri', 'intrus', 'know', 'land', 'last', 'life', 'miss', 'moni', 'mumphlett', 'need', 'nevah', 'night', 'noli', 'panic', 'piano', 'poo', 'prom', 'pumpkin', 'realli', 'right', 'robsten', 'roll', 'sleep', 'sore', 'stewpatti', 'still', 'tell', 'terribl', 'till', 'toe', 'tull', 'twfarley', 'twitter', 'week', 'weekend', 'wendi', 'work', 'ya', 'yeah']

        # Act
        cv, log_model = trainer.train(10, self.testing_data_train, self.testfile) #train on all 20 in testing_trainfile

        # Assert
        self.assertEqual(cv.get_feature_names(), cv_act_feature_names)

    def test_train_log_model(self):
        # Arrange
        act_prob = [0.6608568434115749, 0.3391431565884251]
        # Act
        cv, log_model = trainer.train(10, self.testing_data_train, self.testfile) #train on all 20 in testing_trainfile

        # Assert
        corpus_test = trainer.corpusGenerator('I love ice-cream and the simpsons! Also really hungry for pizza. Off to work now yeah.')
        x = cv.transform(corpus_test).toarray()
        test_prob = log_model.predict_proba(x)[0].tolist()
        self.assertEqual(test_prob, act_prob)

if __name__ == '__main__':
    unittest.main()
