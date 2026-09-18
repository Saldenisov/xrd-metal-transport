//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file SAXSRunAction.cc
/// \brief Implementation of the SAXSRunAction class

#include "SAXSRunAction.hh"

#include "SAXSDetectorConstruction.hh"
#include "SAXSRunActionMessenger.hh"
#include "SAXSScoreMode.hh"

#include "G4AnalysisManager.hh"
#include "G4EmCalculator.hh"
#include "G4Gamma.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Material.hh"
#include "G4ProcessManager.hh"
#include "G4PenelopeOscillatorManager.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4VEmProcess.hh"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <vector>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SAXSRunAction::SAXSRunAction() : G4UserRunAction()
{
  // define the messenger
  fMessenger = new SAXSRunActionMessenger(this);

  // default output filename (can be set through macro)
  fFileName = "output";

  // Create the analysis manager
  fAnalysisManager = G4AnalysisManager::Instance();

  const bool imageCsv = SAXSImageOnly() && std::getenv("SAXS_IMAGE_CSV");
  fAnalysisManager->SetDefaultFileType(imageCsv ? "csv" : "root");
  fAnalysisManager->SetFileName(fFileName);
  if (!imageCsv) fAnalysisManager->SetNtupleMerging(true);  // only for root
  fAnalysisManager->SetVerboseLevel(1);

  if (SAXSImageOnly()) {
    const int pixels = SAXSImagePixels();
    const double pitchUm = SAXSImagePitchUm();
    if (pixels <= 0 || pitchUm <= 0.0) {
      G4Exception("SAXSRunAction", "BadImageGeometry", FatalException,
                  "SAXS_IMAGE_PIXELS and SAXS_IMAGE_PITCH_UM must be positive");
    }
    const double halfWidthMm = 0.0005 * pixels * pitchUm;
    // Detector-hit classes in the same accepted image field of view:
    // direct, one Rayleigh, multiple Rayleigh, any Compton in the sample.
    fAnalysisManager->CreateH1("channels", "Primary photon detector classes",
                               4, -0.5, 3.5);
    fAnalysisManager->CreateH2("image", "Primary photon detector counts",
                               pixels, -halfWidthMm, halfWidthMm,
                               pixels, -halfWidthMm, halfWidthMm);
    return;
  }

  // Creating the SD scoring ntuple
  fAnalysisManager->CreateNtuple("part", "Particle");
  fAnalysisManager->CreateNtupleDColumn("e");
  fAnalysisManager->CreateNtupleDColumn("posx");
  fAnalysisManager->CreateNtupleDColumn("posy");
  fAnalysisManager->CreateNtupleDColumn("posz");
  fAnalysisManager->CreateNtupleDColumn("momx");
  fAnalysisManager->CreateNtupleDColumn("momy");
  fAnalysisManager->CreateNtupleDColumn("momz");
  fAnalysisManager->CreateNtupleDColumn("t");
  fAnalysisManager->CreateNtupleIColumn("type");
  fAnalysisManager->CreateNtupleIColumn("trackID");
  fAnalysisManager->CreateNtupleIColumn("NRi");
  fAnalysisManager->CreateNtupleIColumn("NCi");
  fAnalysisManager->CreateNtupleIColumn("NDi");
  fAnalysisManager->CreateNtupleIColumn("eventID");
  fAnalysisManager->CreateNtupleDColumn("weight");
  fAnalysisManager->FinishNtuple();

  // Creating ntuple for scattering
  fAnalysisManager->CreateNtuple("scatt", "Scattering");
  fAnalysisManager->CreateNtupleIColumn("processID");
  fAnalysisManager->CreateNtupleDColumn("e");
  fAnalysisManager->CreateNtupleDColumn("theta");
  fAnalysisManager->CreateNtupleDColumn("weight");
  fAnalysisManager->CreateNtupleDColumn("e_after");
  fAnalysisManager->FinishNtuple();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

SAXSRunAction::~SAXSRunAction()
{
  delete fMessenger;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SAXSRunAction::BeginOfRunAction(const G4Run*)
{
  // Opt-in cross-section audit for independent CPU/GPU transport benchmarks.
  // The values come from the initialized Geant4 physics list and the actual
  // phantom material, not from a separately maintained attenuation table.
  if (IsMaster() && std::getenv("SAXS_DUMP_XS")) {
    const auto* detector = static_cast<const SAXSDetectorConstruction*>(
      G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    G4EmCalculator calculator;
    std::vector<G4double> energies = {10., 15., 20., 22.162917, 25., 30.};
    if (const char* extra = std::getenv("SAXS_DUMP_XS_EXTRA_KEV")) {
      char* end = nullptr;
      const G4double energy = std::strtod(extra, &end);
      if (end != extra && *end == '\0' && energy > 0.) energies.push_back(energy);
    }
    for (const auto* material : {detector->GetPhantom()->GetMaterial(),
                                 G4Material::GetMaterial("Air")}) {
      for (const auto energy : energies) {
        G4cout << std::setprecision(12) << "SAXS_XS " << material->GetName()
               << " " << energy;
        for (const auto* process : {"phot", "compt", "Rayl"}) {
          G4cout << " " << process << "="
                 << calculator.GetCrossSectionPerVolume(energy * keV, G4Gamma::Gamma(),
                                                        process, material) * mm;
        }
        G4cout << " mm^-1" << G4endl;
      }
    }
    // G4EmCalculator queries model cross sections. Tracking uses each
    // G4VEmProcess lambda table, which can differ slightly after tabulation.
    auto* gammaProcesses = G4Gamma::Gamma()->GetProcessManager()->GetProcessList();
    auto* world = G4LogicalVolumeStore::GetInstance()->GetVolume("WorldLogic");
    for (const auto* volume : {detector->GetPhantom(), world}) {
      const auto* couple = volume->GetMaterialCutsCouple();
      for (const auto energy : energies) {
        G4cout << std::setprecision(12) << "SAXS_TRACK_XS " << volume->GetMaterial()->GetName()
               << " " << energy;
        for (const auto* processName : {"phot", "compt", "Rayl"}) {
          G4double lambda = -1.;
          for (std::size_t index = 0; index < gammaProcesses->size(); ++index) {
            auto* process = (*gammaProcesses)[static_cast<G4int>(index)];
            if (process->GetProcessName() == processName) {
              auto* emProcess = dynamic_cast<G4VEmProcess*>(process);
              if (emProcess) lambda = emProcess->GetLambda(energy * keV, couple,
                                                          std::log(energy * keV)) * mm;
              break;
            }
          }
          G4cout << " " << processName << "=" << lambda;
        }
        G4cout << " mm^-1" << G4endl;
      }
    }
  }
  // Export the *actual* Penelope-2008 shell table used by this physics list.
  // A GPU port must not silently substitute PENELOPE-2006 MC-GPU material data.
  if (IsMaster() && std::getenv("SAXS_DUMP_COMPTON_OSC")) {
    const auto* detector = static_cast<const SAXSDetectorConstruction*>(
      G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    auto* manager = G4PenelopeOscillatorManager::GetOscillatorManager();
    for (const auto* material : {detector->GetPhantom()->GetMaterial(),
                                 G4Material::GetMaterial("Air")}) {
      const auto* table = manager->GetOscillatorTableCompton(material);
      for (std::size_t i = 0; i < table->size(); ++i) {
        auto* oscillator = (*table)[i];
        std::ostringstream line;
        line << std::setprecision(17) << "SAXS_COMPTON_OSC " << material->GetName()
             << " " << i << " " << oscillator->GetOscillatorStrength()
             << " " << oscillator->GetIonisationEnergy() / keV
             << " " << oscillator->GetHartreeFactor();
        G4cout << line.str() << G4endl;
      }
    }
  }
  // open the output file
  if (!fIsFileOpened) {
    fAnalysisManager->OpenFile(fFileName);
    fIsFileOpened = true;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SAXSRunAction::EndOfRunAction(const G4Run* run)
{
  G4int nofEvents = run->GetNumberOfEvent();
  if (nofEvents == 0) return;

  // print
  if (IsMaster()) {
    G4cout << G4endl << "--------------------End of Global Run-----------------------" << G4endl
           << " The run had " << nofEvents << " events";
  }
  else {
    G4cout << G4endl << "--------------------End of Local Run------------------------" << G4endl
           << " The run had " << nofEvents << " events";
  }
  if (fIsFileOpened) {
    fAnalysisManager->Write();
    fAnalysisManager->CloseFile();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SAXSRunAction::SetFileName(const G4String& filename)
{
  // method to set the output filename
  if (filename != "") fFileName = filename;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
