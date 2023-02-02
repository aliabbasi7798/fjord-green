
def carbonEmission(modelSize , uploadRate , downloadRate  , eRouter , eClient, numberClient, time , pArr):
    carbonfactor = 1
    for i in range(numberClient):
        return carbonfactor*energy(modelSize*pArr[i] , uploadRate, downloadRate , eRouter , eClient, numberClient, time , 0.9872 * pArr[i])

def energy(modelSize , uploadRate , downloadRate  , eRouter , eClient, numberClient, time , energyrate):
    return communicationEnergy(modelSize , uploadRate, downloadRate  , eRouter , eClient, numberClient)  + computationEnergy(time , energyrate , numberClient)

def communicationEnergy(modelSize , uploadRate , downloadRate  , eRouter , eClient, numberClient):
    totalEnergy = 0
    for i in range(numberClient):
        totalEnergy = modelSize * (1 /uploadRate + 1/downloadRate)*(eRouter + eClient)
    return totalEnergy

def computationEnergy(time , energyrate , numberofclients):
    totalEnergy = 0
    for i in range(numberofclients):
       totalEnergy += time[i] * energyrate

    return totalEnergy
