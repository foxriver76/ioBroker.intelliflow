/**
 * Intelliflow adapter
 */

/* jshint -W097 */// jshint strict:false
/*jslint node: true */
'use strict';

const utils = require(`@iobroker/adapter-core`);
const np = require(`numjs`);
let adapter;
const clf = [];

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

        // Subscribe to the trigger
        if (task[`trigger`]) {
            adapter.log.info(`Subscribe to trigger ${task[`trigger`]} for ${task[`name-id`]}`);
            adapter.on(task[`trigger`], obj => {
                adapter.log.info(`Triggered prediction of ${task[`name-id`]} by ${task[`trigger`]}`);
            });
        } // endIf

        // and do prediction by interval
        if (task[`interval`]) {
            adapter.log.info(`Create prediction interval (${parseInt(task[`interval`])} seconds) for ${task[`name-id`]}`);
            setInterval(() => {
                adapter.log.info(`Triggered prediction of ${task[`name-id`]} by interval (${task[`interval`]} seconds)`);
            }, parseInt(task[`interval`]) * 1000);
        } // endIf

        // todo: learn when currentCorrectLabel changes
    } // endFor
} // endMain

if (module && module.parent) {
    module.exports = startAdapter;
} else {
    // or start the instance directly
    startAdapter();
} // endElse
