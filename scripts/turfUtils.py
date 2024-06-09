# import vtk
# import paramiko
import numpy as np
from os import getcwd, chdir, path
# from vtkmodules.util.numpy_support import vtk_to_numpy


def selectReader(
    dataType,
    rdr,
):
    extractorFunction = {
        "vtp": rdr.GetOutput().GetPointData(),
        "vtu": rdr.GetOutput().GetCellData(),
        "vts": rdr.GetOutput().GetCellData(),
    }
    return extractorFunction[dataType]

def extractData(
    fName,
    fLoc = None,
    selectedFields = None
):
    initialWD = getcwd()
    if fLoc is not None:
        chdir(fLoc)
    # else:
    #     initialWD = getcwd()
    ## Reads data from the given file name and prepares the data tensor
    fileExtension = fName.split(".")[1]
    # print(fName,fileExtension)
    readers = {
        "vtp": vtk.vtkXMLPolyDataReader(),
        "vtu": vtk.vtkXMLUnstructuredGridReader(),
        "vts": vtk.vtkXMLStructuredGridReader(),
    }
    reader = readers[fileExtension]
    reader.SetFileName(fName)
    reader.Update()
    extractor = selectReader(fileExtension,reader)
    # print(extractor)
    nArrays = extractor.GetNumberOfArrays()
    if selectedFields is None:
        selectedFields = list(range(nArrays))
    # print(nArrays)

    # nArrays = reader.GetOutput().GetCellData().GetNumberOfArrays()
    nCells = reader.GetNumberOfCells()
    # print(nArrays,nCells)
    dataArray = np.zeros((nCells,len(selectedFields)))
    # dataArray = np.zeros((nCells,nArrays))
    # for dataIdx in range(nArrays):
    arrayNames=[]
    dtypes=[]
    for idx, dataIdx in enumerate(selectedFields):
        # print(extractor.GetArray(dataIdx))
        dataArray[:,idx] =  vtk_to_numpy(extractor.GetArray(dataIdx))
        arrayNames.append(extractor.GetArrayName(dataIdx))
        dtypes.append(vtk_to_numpy(extractor.GetArray(dataIdx)).dtype)
        # print(dataIdx,vtk_to_numpy(reader.GetOutput().GetCellData().GetArray(dataIdx)).min(),vtk_to_numpy(reader.GetOutput().GetCellData().GetArray(dataIdx)).max())

    # print(reader.GetNumberOfCells())
    # print(reader.GetNumberOfPoints())
    # print(vtk_to_numpy(reader.GetOutput().GetCellData().GetArray(0)))
    # print((reader.GetOutput().GetCellData().GetNumberOfArrays()))
    # print(reader.GetNumberOfFields())



    chdir(initialWD)
    if fileExtension == 'vts':
        return dataArray , len(selectedFields), arrayNames, dtypes ,vtk_to_numpy(reader.GetOutput().GetFieldData().GetArray(0))[0]
    elif fileExtension == 'vtu':
        return dataArray , len(selectedFields), arrayNames, dtypes
    else:
        raise TypeError("Unknown file type!")
    
def extractAll(
        fName,
        fLoc = None,
):
    initialWD = getcwd()
    if fLoc is not None:
        chdir(fLoc)
    fileExtension = fName.split(".")[1]
    readers = {
        "vtp": vtk.vtkXMLPolyDataReader(),
        "vtu": vtk.vtkXMLUnstructuredGridReader(),
        "vts": vtk.vtkXMLStructuredGridReader(),
    }

    reader = readers[fileExtension]
    reader.SetFileName(fName)
    reader.Update()
    extractor = selectReader(fileExtension,reader)

    nArrays = extractor.GetNumberOfArrays()
    nCells = reader.GetOutput().GetNumberOfCells()


    nCells = reader.GetNumberOfCells()

    arrayNames = []
    dtypes = []
    if fileExtension == "vtu":
        # vtk_to_numpy(reader.GetOutput().GetCells().GetData()).reshape(nCells,-1)
        connectivity = vtk_to_numpy(reader.GetOutput().GetCells().GetConnectivityArray()).reshape(nCells,-1)
        offsets = vtk_to_numpy(reader.GetOutput().GetCells().GetOffsetsArray())[1:]#.reshape(nCells,-1)
        nArrays += connectivity.shape[-1]+1 +1
        dataArray = np.zeros((nCells,nArrays))
        for idx, dataIdx in enumerate(list(range(extractor.GetNumberOfArrays()))):
            dataArray[:,idx] =  vtk_to_numpy(extractor.GetArray(dataIdx))
            arrayNames.append(extractor.GetArrayName(idx))
            dtypes.append(vtk_to_numpy(extractor.GetArray(dataIdx)).dtype)
        dataArray[:,extractor.GetNumberOfArrays()-nArrays:-2] = connectivity
        dtypes.append(vtk_to_numpy(reader.GetOutput().GetCells().GetConnectivityArray()).dtype)
        dtypes.append(vtk_to_numpy(reader.GetOutput().GetCells().GetConnectivityArray()).dtype)
        dtypes.append(vtk_to_numpy(reader.GetOutput().GetCells().GetConnectivityArray()).dtype)
        arrayNames.append("connectivity")
        arrayNames.append("connectivity")
        arrayNames.append("connectivity")
        dataArray[:,-2] = offsets
        dtypes.append(vtk_to_numpy(reader.GetOutput().GetCells().GetOffsetsArray()).dtype)
        arrayNames.append("offsets")
        arrayNames.append("types")
        cellTypes = []
        for idx in range(nCells):
            cellTypes.append(reader.GetOutput().GetCellType(idx))
        # cellTypes = np.array(cellTypes)
        dataArray[:,-1] = np.array(cellTypes)
        dtypes.append(dtypes[-1])

    else:
        dataArray = np.zeros((nCells,nArrays))
        for idx, dataIdx in enumerate(list(range(nArrays))):
            dataArray[:,idx] =  vtk_to_numpy(extractor.GetArray(dataIdx))
            arrayNames.append(extractor.GetArrayName(idx))
            dtypes.append(vtk_to_numpy(extractor.GetArray(dataIdx)).dtype)
        
        
    

    chdir(initialWD)
    if fileExtension == "vts":
        return dataArray , arrayNames , dtypes, vtk_to_numpy(reader.GetOutput().GetFieldData().GetArray(0))[0]
    elif fileExtension == "vtu":
        return dataArray , arrayNames , dtypes
    else:
        raise ValueError(f"File extension {fileExtension} not known!")

def extractPoints(
        fName,
        fLoc = None,
        ):
    initialWD = getcwd()
    if fLoc is not None:
        chdir(fLoc)
    fileExtension = fName.split(".")[1]
    readers = {
        "vtp": vtk.vtkXMLPolyDataReader(),
        "vtu": vtk.vtkXMLUnstructuredGridReader(),
        "vts": vtk.vtkXMLStructuredGridReader(),
    }
    reader = readers[fileExtension]
    reader.SetFileName(fName)
    reader.Update()
    # extractor = selectReader(fileExtension,reader)

    # nArrays = extractor.GetNumberOfArrays()
    # nCells = reader.GetOutput().GetNumberOfCells()

    pts = vtk_to_numpy(reader.GetOutput().GetPoints().GetData()) #points

    chdir(initialWD)
    return pts

def normalize(arr:np.array,method:str):
    allowedMethods = ['minmax', 'zscore', 'maxabs', 'medianIQR', 'power', 'sqrt' , 'unitvector', 'none', 'log', 'log10' ,'spec', 'max', 'min', 'spec2', 'spec3', 'spec4', 'adaptiveIQR']
    if method not in allowedMethods:
        raise ValueError("The method should be one of "+" ".join(allowedMethods)+"!")
    
    if method == 'none':
        return arr, 0, 1
    elif np.std(arr) == 0:  # To handle edge cases like all elements being zero or the same number 
        norm_arr = np.zeros_like(arr) # If all the array elements are same, we have a zero array upon any normalization
        norm_constant1 = np.min(arr)
        norm_constant2 = np.max(arr)
    else:
        if method == 'minmax':
            norm_constant1 = np.min(arr)
            norm_constant2 = np.max(arr) - norm_constant1
            norm_arr = (arr - norm_constant1) / norm_constant2

        elif method == 'zscore':
            norm_constant1 = np.mean(arr)
            norm_constant2 = np.std(arr)
            norm_arr = (arr - norm_constant1) / norm_constant2

        elif method == 'maxabs':
            norm_constant1 = 0
            norm_constant2 = np.max(np.abs(arr))
            norm_arr = arr / norm_constant2

        elif method == 'medianIQR':
            norm_constant1 = np.median(arr)
            norm_constant2 = np.subtract(*np.percentile(arr, [75, 25]))
            norm_arr = (arr - norm_constant1) / norm_constant2

        elif method == 'adaptiveIQR':
            norm_constant1 = np.median(arr)
            norm_constant2 = 0
            lower_end = 20
            upper_end = 79
            increment = 1
            while norm_constant2 == 0:
                upper_end += increment
                try:
                    norm_constant2 = np.subtract(*np.percentile(arr, [upper_end, lower_end]))
                except ValueError:
                    upper_end -= 2*increment
                    increment = increment *0.5
                    upper_end += increment
                    norm_constant2 = np.subtract(*np.percentile(arr, [upper_end, lower_end]))
                    print(upper_end,norm_constant2,arr.min(),arr.max())
            norm_arr = (arr - norm_constant1) / norm_constant2

        elif method == 'power':
            norm_constant1 = 0
            norm_constant2 = 1 / np.std(arr)**0.5
            norm_arr = np.sqrt(np.abs(arr)) * np.sign(arr) * norm_constant2

        elif method == 'unitvector':
            norm_constant1 = 0
            norm_constant2 = np.linalg.norm(arr)
            norm_arr = arr / norm_constant2
        elif method == 'log':
            norm_constant1 = 0   
            norm_constant2 = 1 
            norm_arr = np.log1p(arr)  # applies log(x+1) transformation

        elif method == 'log10':
            norm_constant1 = 0   
            norm_constant2 = 1 
            norm_arr = np.log10(arr+1)  # applies log(x+1) transformation base 10
        
        elif method == 'sqrt':
            norm_constant1 = 0   
            norm_constant2 = 1
            norm_arr = np.sqrt(arr)

        elif method == "spec":
            norm_constant1 = 0   
            norm_constant2 = 1 
            norm_arr = 1/arr
        
        elif method == 'max':
            norm_constant1 = 0   
            norm_constant2 = np.max(arr) 
            norm_arr = arr/norm_constant2

        elif method == 'spec2':
            arr = 1/arr
            norm_constant1 = np.mean(arr)
            norm_constant2 = np.std(arr)
            norm_arr = (arr - norm_constant1) / norm_constant2
        
        elif method == 'spec3':
            norm_constant1 = np.min(arr)
            arr -=norm_constant1
            norm_constant2 = np.max(arr)
            norm_arr = arr/norm_constant2

        elif method == 'spec4':
            norm_constant1 = np.min(arr)
            arr -=norm_constant1
            norm_constant2 = np.linalg.norm(arr)
            norm_arr = arr/norm_constant2

        elif method == 'min':
            norm_constant1 = np.min(arr)
            norm_constant2 = 1
            norm_arr = arr -norm_constant1
            # norm_arr = arr/norm_constant2
            # norm_arr = (arr - norm_constant1) / norm_constant2
            



    return norm_arr, norm_constant1, norm_constant2

    # if np.std(arr)==0:    # To handle edge cases like all elements being zero or the same number 
    #     if method == 'zscore': # Division by zero is undefined, return input array
    #         #   norm_arr = arr
    #           norm_constant1 = 0  
    #           norm_constant2 = 0
    #     elif method == 'minmax':
    #           norm_constant1 = np.min(arr)
    #           norm_constant2 = np.max(arr)
    #           arr = np.zeros_like(arr) # If all the array elements are same, we have a zero array upon min-max normalization

    # else:
    #     if method == 'minmax':
    #         norm_constant1 = np.min(arr)
    #         norm_constant2 = np.max(arr) - norm_constant1
    #         arr = (arr - norm_constant1) / norm_constant2

    #     elif method == 'zscore':
    #         norm_constant1 = np.mean(arr)
    #         norm_constant2 = np.std(arr)
    #         arr = (arr - norm_constant1) / norm_constant2
    # return arr, norm_constant1, norm_constant2

def denormalize(norm_arr, method, norm_constant1, norm_constant2):
    allowedMethods = ['minmax', 'zscore', 'maxabs', 'medianIQR', 'power', 'sqrt', 'unitvector', 'none', 'log', 'spec', 'max', 'min', 'spec2', 'spec3', 'spec4', 'adaptiveIQR']
    if method not in allowedMethods:
        raise ValueError("The method should be one of "+" ".join(allowedMethods)+"!")

    if method == 'none':
        return norm_arr
    elif method == 'minmax':
        arr = norm_arr * norm_constant2 + norm_constant1
    elif method == 'zscore':
        arr = norm_arr * norm_constant2 + norm_constant1
    elif method == 'maxabs':
        arr = norm_arr * norm_constant2
    elif method == 'medianIQR':
        arr = norm_arr * norm_constant2 + norm_constant1
    elif method == 'adaptiveIQR':
        arr = norm_arr * norm_constant2 + norm_constant1
    elif method == 'power':
        arr = ((norm_arr / norm_constant2) ** 2) * np.sign(norm_arr)
    elif method == 'unitvector':
        arr = norm_arr * norm_constant2  + norm_constant1
    elif method == 'log':
        arr = np.expm1(norm_arr)
    elif method == 'log10':
        arr = 10**norm_arr - 1
    elif method == 'sqrt':
        arr = arr**2
    elif method == 'spec':
        arr = 1/norm_arr
    elif method == 'max':
        arr = norm_arr * norm_constant2 + norm_constant1
    elif method == 'min':
        arr = norm_arr * norm_constant2 + norm_constant1
    elif method == 'spec2':
        arr = norm_arr * norm_constant2 + norm_constant1
        arr = 1/arr
    elif method == 'spec3':
        arr = norm_arr * norm_constant2 + norm_constant1
    elif method == 'spec4':
        arr = norm_arr * norm_constant2 + norm_constant1

    return arr

def mkdir_p(sftp, remote_directory):
    """Change to this directory, recursively making new folders if needed.
    Returns True if any folders were created."""
    if remote_directory == '/':
        # absolute path so change directory to root
        sftp.chdir('/')
        return
    if remote_directory == '':
        # top-level relative directory must exist
        return
    try:
        sftp.chdir(remote_directory) # sub-directory exists
    except IOError:
        dirname, basename = path.split(remote_directory.rstrip('/'))
        mkdir_p(sftp, dirname) # make parent directories
        sftp.mkdir(basename) # sub-directory missing, so created it
        sftp.chdir(basename)
        return True

# Keeping this for further reference
# vtk_type_by_numpy_type = {
#         np.uint8: vtk.VTK_UNSIGNED_CHAR,
#         np.uint16: vtk.VTK_UNSIGNED_SHORT,
#         np.uint32: vtk.VTK_UNSIGNED_INT,
#         np.uint64: vtk.VTK_UNSIGNED_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_UNSIGNED_LONG_LONG,
#         np.int8: vtk.VTK_CHAR,
#         np.int16: vtk.VTK_SHORT,
#         np.int32: vtk.VTK_INT,
#         np.int64: vtk.VTK_LONG if vtk.VTK_SIZEOF_LONG == 64 else vtk.VTK_LONG_LONG,
#         np.float32: vtk.VTK_FLOAT,
#         np.float64: vtk.VTK_DOUBLE
#     }

# vtkDataArrayTypes = {
#         'uint8': vtk.vtkUnsignedCharArray(),
#         'uint16': vtk.vtkUnsignedShortArray(),
#         'uint32': vtk.vtkUnsignedIntArray(),
#         'uint64': vtk.vtkUnsignedLongArray() if vtk.VTK_SIZEOF_LONG == 64 else vtk.vtkUnsignedLongLongArray(),
#         'int8': vtk.vtkCharArray(),
#         'int16': vtk.vtkShortArray(),
#         'int32': vtk.vtkIntArray(),
#         'int64': vtk.vtkLongArray() if vtk.VTK_SIZEOF_LONG == 64 else vtk.vtkLongLongArray(),
#         'float32': vtk.vtkFloatArray(),
#         'float64': vtk.vtkDoubleArray()
# }