/*
*   Adaptive Robust Soft Learning Vector Quantization
*
*   References
*   ----------
*    .. [1] Heusinger, M., Raab, C., Schleif, F.M.: Passive concept drift
*    handling via momentum based robust soft learning vector quantization.\
*    In: Vellido, A., Gibert, K., Angulo, C., Martı́n Guerrero, J.D. (eds.)
*    Advances in Self-Organizing Maps, Learning Vector Quantization,
*    Clustering and Data Visualization. pp. 200–209. Springer International
*    Publishing, Cham (2020)
*    .. [2] Sambu Seo and Klaus Obermayer. 2003. Soft learning vector
*    quantization. Neural Comput. 15, 7 (July 2003), 1589-1604
 */

const nj = require(`numjs`);

class RSLVQ {

    constructor(options) {
        this.options = options || {};
        this.prototypesPerClass = this.options.prototypesPerClass || 2;
        this.sigma = this.options.sigma || 1.0;
        this.epsilon = 0.00000001;
        this.initialPrototypes = this.options.initialPrototypes || false;
        this.gamma = this.options.gamma || 0.9;
        this.w = {};
        this.logger = options.logger;
        this.squaredMeanGradient = {};
        this.squaredMeanStep = {};

        // if we already have prototypes init them
        if (this.initialPrototypes) {
            this.validatePrototypes(null, null);
        } // endIf
    } // endConstructor

    async partialFit(X, y) {
        // validate prototypes
        await this.validatePrototypes(X, y);

        // optimize by adadelta
        await this.optimize(X, y);
    } // endPartialFit

    async validatePrototypes(X, y) {
        // initialize prototypes
        if (this.initialPrototypes) {
            // use stored prototypes
            this.w = this.initialPrototypes;
            this.initialPrototypes = null;
        } else if (!this.w[y]) {
            // else we are getting a single label and init to the first datapoint
            for (let i = 0; i < this.prototypesPerClass; i++) {
                if (!this.w[y]) this.w[y] = {};
                this.w[y][i] = X;
            } // endFor
        } // endElse
    } // endValidatePrototypes

    async optimize(X, y) {
        let highestCostCorr;
        let highestCostIncorr;
        let labelCorr;
        let protoCorr;
        let labelIncorr;
        let protoIncorr;

        for (const label in this.w) {
            for (const proto in this.w[label]) {
                const cost = await this.costFun(X, nj.array(this.w[label][proto]));
                if (parseInt(label) === parseInt(y)) {
                    if (highestCostCorr === undefined || cost > highestCostCorr) {
                        // we found new best cost
                        highestCostCorr = cost;
                        labelCorr = label;
                        protoCorr = proto;
                    } // endIf
                } else {
                    // label differs, we need best incorrect to reject it
                    if (highestCostIncorr === undefined || cost > highestCostIncorr) {
                        highestCostIncorr = cost;
                        labelIncorr = label;
                        protoIncorr = proto;
                    } // endIf
                } // endElse
            } // endFor
        } // endFor

        // if nearest proto has not correct label, we have to learn
        if (highestCostCorr < highestCostIncorr) {
            await this.updatePrototype(y, labelIncorr, X, protoIncorr);
            await this.updatePrototype(y, labelCorr, X, protoCorr);
        } // endIf
    } // endOptimize

    async updatePrototype(y, label, x, protoIdx) {
        const dist = nj.array(x).subtract(this.w[label][protoIdx]);

        let gradient;
        const posterior = await this.calcPosterior(protoIdx, x, y, false);
        const posteriorWrtY = await this.calcPosterior(protoIdx, x, y, true);

        if (y === label) {
            gradient = nj.array(posteriorWrtY).subtract(nj.array(posterior)).multiply(dist);
        } else {
            gradient = nj.negative(nj.array(posterior).multiply(dist));
        } // endElse

        // todo: ensure label and protoIdx exist - maybe in validate Proto method
        this.squaredMeanGradient[label][protoIdx] = nj.array(this.gamma).multiply(this.squaredMeanGradient[label][protoIdx]).add(nj.array(1 - this.gamma).multiply(gradient.pow(2)));
        this.logger.warn(`gradient: ` + gradient);

        const step = nj.sqrt(nj.array(this.squaredMeanStep[label][protoIdx]).add(nj.array(this.epsilon)).divide(nj.array(this.squaredMeanStep[label][protoIdx]).add(nj.array(this.epsilon)))).multiply(gradient);

        this.squaredMeanStep[label][protoIdx] = nj.array(this.gamma).multiply(nj.array(this.squaredMeanStep[label][protoIdx])).add(nj.array(1 - this.gamma).multiply(step.pow(2)));

        // finally update the prototype
        this.w[label][protoIdx] = this.w[label][protoIdx].add(step);
    } // endUpdatePrototype

    async predict(X) {
        let bestLabel;
        let bestCost;
        for (const label in this.w) {
            for (const proto in this.w[label]) {
                const w = nj.array(this.w[label][proto]);
                const currentCost = await this.costFun(X, w);
                // check if current cost greater, because maximization problem
                if (bestCost === undefined || currentCost > bestCost) {
                    bestCost = currentCost;
                    bestLabel = label;
                } // endIf
            } // endFor
        } // endFor
        return bestLabel;
    } // endPredict

    async costFun(x, w) {
        let d = x.subtract(nj.array(w));
        d = d.T.dot(d);
        return d.divide(2 * this.sigma).multiply(-1).get(0);
    } // endCostFun

    async calcPosterior(protoIdx, x, y, respectY) {
        const fs = [];
        if (!respectY) {
            for (const label in this.w) {
                for (const proto in this.w[label]) {
                    const w = nj.array(this.w[label][proto]);
                    fs.push(await this.costFun(x, w));
                } // endFor
            } // endFor
        } else {
            for (const label in this.w) {
                // if y is given, we are calculating the posterior w.r.t. y
                if (parseInt(label) !== y) continue;
                for (const proto in this.w[label]) {
                    const w = nj.array(this.w[label][proto]);
                    fs.push(await this.costFun(x, w));
                } // endFor
            } // endFor
        } // endElse
        const fsMax = nj.max(fs);

        const s = [];

        for (const f of fs) {
            s.push(Math.exp(f - fsMax));
        } // endFor

        const sSum = nj.array(s).sum();

        // we need to broadcast fsmax and sSum because nj array currently does not support this
        let broadcastedFsMax = [];
        let broadcastedsSum = [];

        for (const i in this.w[y][protoIdx]) {
            broadcastedFsMax[i] = fsMax;
            broadcastedsSum[i] = sSum;
        } // endFor

        broadcastedsSum = nj.array(broadcastedsSum);
        broadcastedFsMax = nj.array(broadcastedFsMax);

        const cosFun = await this.costFun(x, nj.array(this.w[y][protoIdx]).subtract(broadcastedFsMax));
        // now broadcast cosFun
        let broadcastedCosFun = [];
        for (const i in this.w[y][protoIdx]) {
            broadcastedCosFun[i] = cosFun;
        } // endFor

        broadcastedCosFun = nj.array(broadcastedCosFun);

        return nj.exp(broadcastedCosFun).divide(broadcastedsSum);
    } // endCalcPosterior

} // endClassRSLVQ

module.exports = RSLVQ;