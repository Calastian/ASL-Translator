import preprocess as pp 

pp.makePreProcessedData(glossCSVPath='../ASL_Citizen/splits/val.csv',
                        videoFolderPath='../ASL_Citizen/videos/',
                        outputCSVFilePath='../src/data/small_val.csv',
                        filterCSVPath='../docs/Key_ASL.csv', chunkSize=25, startIndex=0,stopIndex=79
                        )


# makePreProcessedData('archive/ASL_Citizen/splits/train.csv','archive/ASL_Citizen/videos/',
#                   'processedData/output.csv', 'Key_ASL.csv', chunkSize=5, startIndex=78)