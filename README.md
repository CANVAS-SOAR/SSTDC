# SSTDC The following information is relative to working on the repo on the lab machine

The SSTDC repo is located in ~/Documents/SSTDC/net_classifier and has been renamed net_classifier

to run the classifier, first run this command to activate the tensorflow virtualenv for gpus
> source ~/tensorflow/bin/activate

Then navigate to the net_classifier folder
> cd Documents/SSTDC/net_classifier

Then you should be able to simply run the classifier.py
> python3 classifier.py


To deactivate the tensorflow virtualenv
> deactivate

Don't forget to push changes to the master branch on github to keep everyone up to date

To use tensorboard to view the network visually, and view metrics vs iterations
cd to net_classifier
> cd Documents/SSTDC/net_classifier # or appropriate folder, where logs is below it
then enter
> tensorboard --logdir="logs"
you should get a display that contains a link, go to the link (likely 0.0.0.0:6006 or something)
***
Note that I have not gotten tensorboard to work through ssh.
I believe you will need to copy the event file from logs/ to a local machine where tensorflow is installed, go to the appropriate directory, and use the tensorboard command below with the appropriate name of the folder with the eventfile. I have not yet had luck copying the file over to the local machine through ssh but I think this would be the best way to solve the problem

Also, using tensoboard can be good for debugging. The cross_correlation is essentially the loss function, which should decrease with time. If it does not possibly try playing with the step size in gradient descent
***


