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
                write: false,
                def: false
            },
            native: {}
        });

        // Create object to store prototypes of RSLVQ
        adapter.setObjectNotExists(`${task[`name-id`]}.prototypes`, {
            type: `state`,
            common: {
                name: `Prototypes of RSLVQ for ${task[`name-id`]}`,
                role: `list`,
                type: `json`,
                read: true,
                write: false,
                def: false
            },
            native: {}
        });

        // Create datapoints to learn, by copying the label
        try {
            const labelObj = await adapter.getForeignObjectAsync(task[`label`]);

            adapter.setObjectNotExists(`${task[`name-id`]}.currentCorrectLabel`, {
                type: labelObj.type,
                common: {
                    name: `Current Correct Label of ${task[`name-id`]}`,
                    role: labelObj.common.role,
                    type: labelObj.common.type,
                    read: true,
                    write: true
                },
                native: labelObj.native
            });
        } catch (e) {
            adapter.log.warn(`Error on copying label object, using default: ${e}`);
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
        } // endTryCatch

        // get name of the features once - they won't change during runtime
        task.featureNames = await getFeatureNames(task);
        adapter.log.info(`Got feature names for ${task[`name-id`]}: ${JSON.stringify(task.featureNames)}`);

        const initialPrototypesState = await adapter.getStateAsync(`${task[`name-id`]}.prototypes`);

        task.classifier = initialPrototypesState && initialPrototypesState.val ? new RSLVQ({initialPrototypes: initialPrototypesState.val}) : new RSLVQ();

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
            let y;

            if (typeof state.val === `boolean`) {
                y = state.val ? 1 : 0;
            } else if (typeof state.val !== `number`) {
                adapter.log.warn(`${state.val} is not boolean and not number - did not learn`);
                return;
            } // endElseIf

            adapter.log.info(`Using feature set to learn label ${y} to ${task[`name-id`]}: ${featureSet}`);

            task.classifier.partialFit(featureSet, y);
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

    // todo: predict via rslvq

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
