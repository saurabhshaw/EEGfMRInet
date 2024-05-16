# -*- coding: utf-8 -*-
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
import shap
import sklearn

######################## Function Definitions ########################

def classify_NN(X_train_scale, X_test_scale, YY_train_labels, YY_test_labels, X_weights, shap_value_numPoints, compute_regression , compute_SHAP):
    
    # Define NN layers:
    model = Sequential()
    model.add(Dense(input_dim=len(X_train_scale[1]), output_dim=500))
    model.add(Activation('relu'))
    # model.add(BatchNormalization())
    
    if compute_regression:
        model.add(Dense(input_dim=500, output_dim=1)) # was 1 for regression
        model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])

    else:      
        model.add(Dense(input_dim=500, output_dim=2)) # was 1 for regression
        model.add(Activation('relu'))          
        model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
    
    model.summary()    
    model.fit(X_train_scale, YY_train_labels,
              batch_size=10, epochs=10, verbose=2,
              validation_data=(X_test_scale, YY_test_labels))
              # sample_weight=np.ravel(X_weights))
    
    test_scores = model.evaluate(X_test_scale,YY_test_labels)
    training_scores = model.evaluate(X_train_scale,YY_train_labels)
    # print(model.predict(X_train_scale, batch_size=1))
    
    ## Identify features based on SHAP:
    if compute_SHAP:   
        [shap_explainer, shap_values, X_test_scale_shap_values] = SHAPdeepexplainer(model, X_train_scale, X_test_scale, shap_value_numPoints)
        
    else:
        shap_explainer = []
        shap_values = []
        
    return test_scores, training_scores, shap_explainer, shap_values, model
        
        

def classify_SVM(X_train_scale, X_test_scale, YY_train_labels, YY_test_labels, X_weights, kernel_input, shap_value_numPoints, compute_regression , compute_SHAP): 
    ## Run SVM
    if compute_regression:
        model_SVM = sklearn.svm.SVR(kernel=kernel_input, gamma = 'scale')
        model_SVM.fit(X_train_scale, YY_train_labels)
        Yhat_test = model_SVM.predict(X_test_scale)
        Yhat_train = model_SVM.predict(X_train_scale)
     
        test_scores = mean_squared_error(Yhat_test,YY_test_labels)
        training_scores = mean_squared_error(Yhat_train,YY_train_labels)
    
    else:    
        model_SVM = sklearn.svm.SVC(kernel=kernel_input, gamma = 'scale', probability=True)
        model_SVM.fit(X_train_scale, YY_train_labels)
        Yhat_test = model_SVM.predict(X_test_scale)
        Yhat_train = model_SVM.predict(X_train_scale)
        
        test_scores = accuracy_score(YY_test_labels,Yhat_test)
        training_scores = accuracy_score(YY_train_labels,Yhat_train)
        #test_scores_SVM = np.count_nonzero(Yhat_test == Y[x_test_vect])/len(Y[x_test_vect])
        #training_scores_SVM = np.count_nonzero(Yhat_train == Y[x_train_vect])/len(Y[x_train_vect])
            
    # Use Kernel SHAP to explain test set predictions
    if compute_SHAP:
        [shap_explainer, shap_values, X_test_scale_shap_values] = SHAPkernelexplainer(model_SVM, X_train_scale, X_test_scale, shap_value_numPoints, compute_regression)
    else:
        shap_explainer = []
        shap_values = []
        
    return test_scores, training_scores, shap_explainer, shap_values, model_SVM


def SHAPdeepexplainer(model, X_train_scale, X_test_scale, shap_value_numPoints):
    # print the JS visualization code to the notebook
    # shap.initjs()
    
    
    # Select a set of background examples to take an expectation over
    background = X_train_scale[np.random.choice(X_train_scale.shape[0], shap_value_numPoints, replace=False)]
    X_test_scale_shap_values = np.random.choice(X_test_scale.shape[0], shap_value_numPoints, replace=False)
    
    # Explain predictions of the model on four images
    shap_explainer = shap.DeepExplainer(model, background)
    # ...or pass tensors directly
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values = shap_explainer.shap_values(X_test_scale[X_test_scale_shap_values])
    
    ## Plot the feature attributes for Neural Network:
    #shap.summary_plot(shap_values_NN, X_test_scale,plot_type = "violin")
    #shap.force_plot(shap_explainer_NN.expected_value[0], shap_values_NN[0], X_test_scale, link="logit")
    ## shap.image_plot(shap_values_NN, -X_test_scale[1:5])
    
    ## Plot the feature attributions
    # shap.force_plot(shap_explainer.expected_value[0], shap_values[0], X_test_scale, link="logit")
    # shap.image_plot(shap_values, -X_test_scale[1:5])

    return shap_explainer, shap_values, X_test_scale_shap_values


def SHAPkernelexplainer(model_SVM, X_train_scale, X_test_scale, shap_value_numPoints, compute_regression):
    
    X_test_scale_shap_values = np.random.choice(X_test_scale.shape[0], shap_value_numPoints, replace=False)
    
    if compute_regression:        
        # For KernelExplainer in Regression problems, do not use link='logit', otherwise will throw ValueError: Input contains NaN, infinity or a value too large for dtype('float64'). 
        shap_explainer = shap.KernelExplainer(model_SVM.predict, shap.kmeans(X_train_scale,10))
        shap_values = shap_explainer.shap_values(X_test_scale[X_test_scale_shap_values], nsamples="auto", l1_reg = "aic")

    else:
        shap_explainer = shap.KernelExplainer(model_SVM.predict_proba, shap.kmeans(X_train_scale,10), link="logit")
        shap_values = shap_explainer.shap_values(X_test_scale[X_test_scale_shap_values], nsamples="auto", l1_reg = "aic")
    
    ## Plot the feature attributes:
    #shap.summary_plot(shap_values_SVM, X_test_scale,plot_type = "violin")
    #shap.force_plot(shap_explainer_SVM.expected_value[0], shap_values_SVM[0], X_test_scale[0], link="logit")

    return shap_explainer, shap_values, X_test_scale_shap_values