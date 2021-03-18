from afm.afm_analysis import AFManalysis

## gwyddion actions:
##       required:
##           -use "Level data by mean plane subtraction"
##           -use "Shift minimum data value to zero"
##           -save file as .xyz with Precision set to 10
##       recommended:
##           -use "Correct horizontal scars (strokes)"
##           -use "Remove polynomial background"

if __name__ == "__main__":
    ## parameters to change
    ## file location
    directory = "NP7593"
    fileName = "NP7593_1mum_Mitte2.xyz"
    pathToDirectory = None

    ## physical image Dimensions (for example 1µm x 1µm)
    imageDimensions = (1, 1)        # (µm, µm)
    maximumQDdiameter = 100         # nm

    ## selecting the properties that should be displayed
    showDensity                     = True
    plotHeightDistribution          = True
    plotFWHMDistribution            = True
    plotAspectRatioDistribution     = True
    plotOriginalData                = True
    plotOriginalData3D              = True
    plotFilteredData                = True
    plotFilteredData3D              = True





    ## running the main program using the parameters above
    run = AFManalysis(directory=directory, fileName=fileName, pathToDirectory=pathToDirectory)
    run.load_data(imageDimensions=imageDimensions)
    run.local_max(maxQDSize=maximumQDdiameter)
    run.calculate_properties()

    if plotOriginalData:
        run.plot_data(useUnprocessed=True)
    if plotOriginalData3D:
        run.plot_data(useUnprocessed=True, style="3d")
    if plotFilteredData:
        run.plot_data(useUnprocessed=False)
    if plotFilteredData3D:
        run.plot_data(useUnprocessed=False, style="3d")
    if showDensity:
        run.density()
    if plotHeightDistribution:
        run.height_distribution()
    if plotFWHMDistribution:
        run.diameter_distribution()
    if plotAspectRatioDistribution:
        run.aspectratio_distribution()