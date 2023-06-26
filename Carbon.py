
def carbonEmission(modelSize , uploadRate , downloadRate  , eRouter , eClient, numberClient, numberEpoches , frequency , numberofsmaples , workload , pArr , energyC , carbonIntensity):
    carbonfactor = carbonIntensity
    comuCarbon , compCarbon = 0 , 0
    computationparam = (numberEpoches * numberofsmaples * workload) / frequency
    for i in range(numberClient):

        comutemp , comptemp = energy(modelSize * (pArr[i])* (pArr[i]), uploadRate, downloadRate, eRouter,
                                     eClient, (pArr[i])* (pArr[i])*computationparam, energyC[i])

        comuCarbon += comutemp * carbonfactor[i]
        compCarbon += comptemp * carbonfactor[i]

    return comuCarbon ,compCarbon
def energy(modelSize , uploadRate , downloadRate  , eRouter , eClient, time , energyrate):
    return communicationEnergy(modelSize , uploadRate, downloadRate  , eRouter , eClient), computationEnergy(time , energyrate)

def communicationEnergy(modelSize , uploadRate , downloadRate  , eRouter , eClient):

    totalEnergy = modelSize * (1 /uploadRate + 1/downloadRate)*(eRouter + eClient)
    return totalEnergy

def computationEnergy(time , energyrate):
    totalEnergy = time * energyrate

    return totalEnergy
