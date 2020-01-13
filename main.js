/**
 * Intelliflow adapter
 */

/* jshint -W097 */// jshint strict:false
/*jslint node: true */
'use strict';

const utils = require(`@iobroker/adapter-core`);
const nj = require(`numjs`);
const RSLVQ = require(`${__dirname}/lib/rslvq`);

let adapter;

function startAdapter(options) {
    options = options || {};
    options = {...options, ...{name: `intelliflow`}};

    adapter = new utils.Adapter(options);

    adapter.on(`unload`, callback => {
        try {
            adapter.log.info(`[END] Stopping Intelliflow adapter...`);
            callback();
        } catch (e) {
            callback();
        } // endTryCatch
    });

    adapter.on(`ready`, async () => {
        main();
    });

    adapter.on(`stateChange`, async (id, state) => {
        if (!id || !state || state.ack) return;
        adapter.log.debug(`[STATE] Changed ${id} to ${state.val}`);
    });

    return adapter;
} // endStartAdapter


async function main() {
    // At first let's describe all internal states
    adapter.subscribeStates(`*`);

    // now lets iterate over all of our tasks
    for (const task of adapter.config.tasks) {
        adapter.log.info(`Managing task: ${JSON.stringify(task)}`); // debug later

        // create the prediction output label
        adapter.setObjectNotExists(`${task[`name-id`]}.prediction`, {
            type: `state`,
            common: {
                name: `Prediction of ${task[`name-id`]}`,
                role: `label`,
                type: `boolean`,
                read: true,
                write: false
            },
            native: {}
        });

        // Create object to store prototypes of RSLVQ
        adapter.setObjectNotExists(`${task[`name-id`]}.prototypes`, {
            type: `state`,
            common: {
                name: `Prototypes of RSLVQ for ${task[`name-id`]}`,
                role: `json`,
                type: `object`,
                read: true,
                write: false
            },
            native: {}
        });

        // Create datapoints to learn, label should be numeric
        adapter.setObjectNotExists(`${task[`name-id`]}.currentCorrectLabel`, {
            type: `state`,
            common: {
                name: `Current Correct Label of ${task[`name-id`]}`,
                role: `label`,
                type: `number`,
                read: true,
                write: true
            },
            native: {}
        });

        // get name of the features once - they won't change during runtime
        task.featureNames = await getFeatureNames(task);
        adapter.log.info(`Got feature names for ${task[`name-id`]}: ${JSON.stringify(task.featureNames)}`);

        const initialPrototypesState = await adapter.getStateAsync(`${task[`name-id`]}.prototypes`);
        let initialPrototypes;

        // prepare stored prototypes
        if (initialPrototypesState && initialPrototypesState.val) {
            initialPrototypes = JSON.parse(initialPrototypesState.val);
            for (const label in initialPrototypes.prototypes) {
                for (const proto in initialPrototypes.prototypes[label]) {
                    if (typeof initialPrototypes.prototypes[label][proto] !== `object`) {
                        initialPrototypes.prototypes[label][proto] = JSON.parse(initialPrototypes.prototypes[label][proto]);
                    } // endIf
                    if (typeof initialPrototypes.squaredMeanGradient[label][proto] !== `object`) {
                        initialPrototypes.squaredMeanGradient[label][proto] = JSON.parse(initialPrototypes.squaredMeanGradient[label][proto]);
                    } // endIf
                    if (typeof initialPrototypes.squaredMeanStep[label][proto] !== `object`) {
                        initialPrototypes.squaredMeanStep[label][proto] = JSON.parse(initialPrototypes.squaredMeanStep[label][proto]);
                    } // endIf
                } // endFor
            } // endFor
        } // endIf

        const options = {
            logger: adapter.log,
            initialPrototypes: initialPrototypesState && initialPrototypesState.val ? initialPrototypes : undefined,
            learningRate: adapter.config.rslvqLearningRate,
            sigma: adapter.config.rslvqSigma,
            gamma: adapter.config.rslvqDecayRate,
            beta1: adapter.config.rslvqBeta1,
            beta2: adapter.config.rslvqBeta2,
            gradientOptimizer: adapter.config.selectedClassifier
        };

        task.classifier = new RSLVQ(options);

        if (initialPrototypesState && initialPrototypesState.val) {
            adapter.log.info(`Successfully loaded prototypes for ${task[`name-id`]}: ${JSON.stringify(task.classifier.w)}`);
        } // endIf

        // Subscribe to the trigger
        if (task[`trigger`]) {
            adapter.log.info(`Subscribe to trigger ${task[`trigger`]} for ${task[`name-id`]}`);

            adapter.subscribeForeignStates(task[`trigger`]);

            adapter.on(`stateChange`, id => {
                if (!id || !matchWildcard(id, task[`trigger`])) return;
                adapter.log.debug(`Triggered prediction of ${task[`name-id`]} by ${task[`trigger`]} triggered by ${id}`);
                doPrediction(task);
            });
        } // endIf

        // and do prediction by interval
        if (task[`interval`]) {
            adapter.log.info(`Create prediction interval (${parseInt(task[`interval`])} seconds) for ${task[`name-id`]}`);
            setInterval(() => {
                adapter.log.debug(`Triggered prediction of ${task[`name-id`]} by interval (${task[`interval`]} seconds)`);
                doPrediction(task);
            }, parseInt(task[`interval`]) * 1000);
        } // endIf

        // learn when currentCorrectLabel changes
        adapter.on(`stateChange`, async (id, state) => {
            if (!id || id !== `${adapter.namespace}.${task[`name-id`]}.currentCorrectLabel`) return;

            // get feature set
            const featureSet = await getFeatures(task);
            const y = parseInt(state.val);

            if (isNaN(y)) {
                adapter.log.warn(`Cannot learn ${state.val} for ${task[`name-id`]} - not a number`);
                return;
            } // endIf

            adapter.log.info(`Using feature set to learn label ${y} to ${task[`name-id`]}: ${featureSet}`);
            try {
                await task.classifier.partialFit(featureSet, y);
            } catch (e) {
                adapter.log.warn(`Error fitting data to classifier: ${e}`);
            }

            adapter.log.info(`New prototypes for ${task[`name-id`]}: ${JSON.stringify(task.classifier.w)}`);

            // store all necessary information
            const storeObj = {};
            storeObj.prototypes = task.classifier.w;
            storeObj.squaredMeanGradient = task.classifier.squaredMeanGradient;
            storeObj.squaredMeanStep = task.classifier.squaredMeanStep;

            // after learning store the new prototypes
            adapter.setState(`${task[`name-id`]}.prototypes`, JSON.stringify(storeObj), true);
        });
    } // endFor
} // endMain

/* Internals */
async function getFeatureNames(task) {
    const enumSplitted = task.enum.split(`.`);
    const enumType = enumSplitted[0] === `enum` ? enumSplitted[1] : enumSplitted[0];
    const enumName = enumSplitted[0] === `enum` ? enumSplitted[2] : enumSplitted[1];
    const enums = await adapter.getEnumAsync(enumType);
    const featureNames = enums.result[`enum.${enumType}.${enumName.toLowerCase()}`].common.members;
    return featureNames;
} // endGetFeatureNames

async function doPrediction(task) {
    const featureSet = await getFeatures(task);

    adapter.log.debug(`Using feature set for prediction of ${task[`name-id`]}: ${featureSet}`);
    try {
        const y = await task[`classifier`].predict(featureSet);
        adapter.setState(`${task[`name-id`]}.prediction`, y, true);
        adapter.log.debug(`Predicted label ${y} for ${task[`name-id`]}`);
    } catch (e) {
        adapter.log.warn(`Error predicting label for feature set ${featureSet}: ${e}`);
    } // endCatch
} // endDoPrediction

async function getFeatures(task) {
    // We have to get all values
    const featureSet = [];

    for (const featureName of task.featureNames) {
        let feature = await adapter.getForeignStateAsync(featureName);
        feature = feature.val;

        // preprocess boolean
        if (typeof feature === `boolean`) {
            feature = feature ? 1 : 0;
        } else if (typeof feature !== `number`) {
            adapter.log.warn(`${featureName} is not boolean and not number - skip this feature`);
            continue;
        } // endElseIf

        featureSet.push(feature);
    } // endFor

    return nj.array(featureSet);
} // endGetFeatures

function matchWildcard(str, rule) {
    const escapeRegex = (str) => str.replace(/([.*+?^=!:${}()|\[\]\/\\])/g, `\\$1`);
    return new RegExp(`^${rule.split(`*`).map(escapeRegex).join(`.*`)}$`).test(str);
} // endMatchWildcard

if (module && module.parent) {
    module.exports = startAdapter;
} else {
    // or start the instance directly
    startAdapter();
} // endElse
