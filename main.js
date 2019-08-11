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
                type: `json`,
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
            for (const label in initialPrototypes) {
                for (const proto in initialPrototypes[label]) {
                    if (typeof initialPrototypes[label][proto] !== `object`) {
                        initialPrototypes[label][proto] = JSON.parse(initialPrototypes[label][proto]);
                    } // endIf
                } // endFor
            } // endFor
        } // endIf

        task.classifier = initialPrototypesState && initialPrototypesState.val ? new RSLVQ({
            logger: adapter.log,
            initialPrototypes: initialPrototypes
        }) : new RSLVQ({logger: adapter.log});

        if (initialPrototypesState && initialPrototypesState.val) {
            adapter.log.info(`Successfully loaded prototypes for ${task[`name-id`]}: ${JSON.stringify(task.classifier.w)}`);
        } // endIf

        // Subscribe to the trigger
        if (task[`trigger`]) {
            adapter.log.info(`Subscribe to trigger ${task[`trigger`]} for ${task[`name-id`]}`);

            adapter.subscribeForeignStates(task[`trigger`]);

            adapter.on(`stateChange`, id => {
                if (!id || id !== task[`trigger`]) return;
                adapter.log.info(`Triggered prediction of ${task[`name-id`]} by ${task[`trigger`]}`);
                doPrediction(task);
            });
        } // endIf

        // and do prediction by interval
        if (task[`interval`]) {
            adapter.log.info(`Create prediction interval (${parseInt(task[`interval`])} seconds) for ${task[`name-id`]}`);
            setInterval(() => {
                adapter.log.info(`Triggered prediction of ${task[`name-id`]} by interval (${task[`interval`]} seconds)`);
                doPrediction(task);
            }, parseInt(task[`interval`]) * 1000);
        } // endIf

        // learn when currentCorrectLabel changes
        adapter.on(`stateChange`, async (id, state) => {
            if (!id || id !== `${adapter.namespace}.${task[`name-id`]}.currentCorrectLabel`) return;

            // get feature set
            const featureSet = await getFeatures(task);
            const y = state.val;

            if (typeof y !== `number`) {
                adapter.log.warn(`${state.val} is not a number - did not learn`);
                return;
            } // endElseIf

            adapter.log.info(`Using feature set to learn label ${y} to ${task[`name-id`]}: ${featureSet}`);

            await task.classifier.partialFit(featureSet, y);
            adapter.log.info(`New prototypes for ${task[`name-id`]}: ${JSON.stringify(task.classifier.w)}`);

            // after learning store the new prototypes
            adapter.setState(`${task[`name-id`]}.prototypes`, JSON.stringify(task.classifier.w), true);
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

    adapter.log.info(`Using feature set for prediction of ${task[`name-id`]}: ${featureSet}`);

    const y = await task[`classifier`].predict(featureSet);
    adapter.setState(`${task[`name-id`]}.prediction`, y, true);
    adapter.log.info(`Predicted label ${y} for ${task[`name-id`]}`);
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

if (module && module.parent) {
    module.exports = startAdapter;
} else {
    // or start the instance directly
    startAdapter();
} // endElse
