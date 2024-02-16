From WordTagger import *
    
def Main():

    #Read Data
    doc = "../data/en_gum-ud-test.conllu"
    
    #1 type of tagging using pos
    pred = pos_tagging(test_data)
    
    #Evaluate accuracy
    model_acc = accuracy(test_data, pred)
    print("Accuracy:", model_acc)

    #2nd type of tagging using dependency trees
    dep_pred = dependency_tree(test_data)
    
    #Evaluate result
    uas_metric = uas(test_data, dep_pred)
    las_metric = las(test_data, dep_pred)
    print("UAS:", uas_metric)
    print("LAS:", las_metric)


#Run Main
if __name__ == "__main__":
    main() 