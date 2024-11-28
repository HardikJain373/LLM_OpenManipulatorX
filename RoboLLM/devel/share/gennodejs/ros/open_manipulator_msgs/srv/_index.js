
"use strict";

let SetActuatorState = require('./SetActuatorState.js')
let GetKinematicsPose = require('./GetKinematicsPose.js')
let GetJointPosition = require('./GetJointPosition.js')
let SetJointPosition = require('./SetJointPosition.js')
let SetDrawingTrajectory = require('./SetDrawingTrajectory.js')
let SetKinematicsPose = require('./SetKinematicsPose.js')

module.exports = {
  SetActuatorState: SetActuatorState,
  GetKinematicsPose: GetKinematicsPose,
  GetJointPosition: GetJointPosition,
  SetJointPosition: SetJointPosition,
  SetDrawingTrajectory: SetDrawingTrajectory,
  SetKinematicsPose: SetKinematicsPose,
};
