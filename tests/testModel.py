import preprocess as pp 

pp.makePreProcessedData(glossCSVPath='../ASL_Citizen/splits/train.csv',
                        videoFolderPath='../ASL_Citizen/test_videos/',
                        outputCSVFilePath='processedData/output.csv',
                        filterCSVPath='Key_ASL.csv',
                        )


