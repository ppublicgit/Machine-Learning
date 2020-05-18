import numpy as np


def conf_mat(predictions, targets):
        """Confusion matrix"""
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            predictions = np.where(predictions>0.5,1,0)
        else:
            # 1-of-N encoding
            predictions = np.argmax(predictions,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(predictions==i,1,0)*np.where(targets==j,1,0))

        print("Confusion matrix is:")
        print(cm)
        print(f"Percentage Correct: {np.trace(cm)/np.sum(cm)*100}")
        return cm
