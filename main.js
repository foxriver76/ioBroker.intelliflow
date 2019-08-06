/**
 * Intelliflow adapter
 */

/* jshint -W097 */// jshint strict:false
/*jslint node: true */
'use strict';

const utils = require(`@iobroker/adapter-core`);
const np = require(`numjs`);
let adapter;

function startAdapter(options) {
    options = options || {};
    options = {...options, ...{name: `intelliflow`}};

    adapter = new utils.Adapter(options);

    adapter.on(`unload`, callback => {
        try {
            adapter.log.info(`[END] Stopping Intelliflow adapter...`);
            adapter.setState(`info.connection`, false, true);
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
    adapter.subscribeStates(`*`);
} // endMain

if (module && module.parent) {
    module.exports = startAdapter;
} else {
    // or start the instance directly
    startAdapter();
} // endElse
