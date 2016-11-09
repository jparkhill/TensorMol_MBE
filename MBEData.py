# MBE data deals with everything you needs to generate MBE data for training and testing

class MBE():
	def __init__(self, MSet_=None, Dig_=None, Name_=None, MxTimePerElement_=7200):
		self.path = "./trainsets/"
                self.suffix = ".pdb"
                self.set = MSet_
                self.dig = Dig_
                self.CurrentElement = None # This is a mode switch for when TensorData provides training data.
                self.SamplesPerElement = []
                self.AvailableElements = []
                self.AvailableDataFiles = []
                self.NTest = 0  # assgin this value when the data is loaded
                self.TestRatio = 0.2 # number of cases withheld for testing.
                self.MxTimePerElement=MxTimePerElement_
                self.MxMemPerElement=8000 # Max Array for an element in MB

                self.NTrain = 0
                self.ScratchState=None
                self.ScratchPointer=0 # for non random batch iteration.
                self.scratch_inputs=None
                self.scratch_outputs=None
                self.scratch_test_inputs=None # These should be partitioned out by LoadElementToScratch
                self.scratch_test_outputs=None
                # Ordinarily during training batches will be requested repeatedly
                # for the same element. Introduce some scratch space for that.
                if (not os.path.isdir(self.path)):
                        os.mkdir(self.path)
                if (Name_!= None):
                        self.name = Name_
                        self.Load()
                        self.QueryAvailable() # Should be a sanity check on the data files.
                        return
                elif (MSet_==None or Dig_==None):
                        raise Exception("I need a set and Digester if you're not loading me.")
                self.name = ""



		
