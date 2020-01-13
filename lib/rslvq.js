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
        this.gradientOptimizer = this.options.gradientOptimizer || `rslvq-adadelta`;
        this.prototypesPerClass = this.options.prototypesPerClass ? parseInt(this.options.prototypesPerClass) : 2;
        this.sigma = this.options.sigma ? parseFloat(this.options.sigma) : 1.0;
        this.beta1 = this.options.beta1 ? parseFloat(this.options.beta1) : 0.9;
        this.beta2 = this.options.beta2 ?  parseFloat(this.options.beta2) : 0.999;
        this.learningRate = this.options.learningRate ? parseFloat(this.options.learningRate) : 0.001;
        this.epsilon = 0.00000001;
        this.initialPrototypes = this.options.initialPrototypes || false;
        this.gamma = this.options.gamma ? parseFloat(this.options.gamma) : 0.9;
        this.w = {};
        this.logger = options.logger;
        this.squaredMeanGradient = {};
        this.squaredMeanStep = {};

        if (this.gradientOptimizer === `rslvq-adadelta`) {
            this.updatePrototype = this.updatePrototypeAdadelta;
        } else if (this.gradientOptimizer === `rslvq-adamax`) {
            this.updatePrototype = this.updateProtoypeAdamax;
        } else {
            throw `unknown gradient optimizer: ${this.gradientOptimizer}`;
        } // endElse

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
            this.w = this.initialPrototypes.prototypes;
            this.squaredMeanStep = this.initialPrototypes.squaredMeanStep;
            this.squaredMeanGradient = this.initialPrototypes.squaredMeanGradient;
            this.initialPrototypes = null;
        } else if (!this.w[y]) {
            // else we are getting a single label and init to the first datapoint
            for (let i = 0; i < this.prototypesPerClass; i++) {
                if (!this.w[y]) this.w[y] = {};
                this.w[y][i] = X;
                if (!this.squaredMeanStep[y]) this.squaredMeanStep[y] = {};
                this.squaredMeanStep[y][i] = nj.zeros(X.tolist().length);
                if (!this.squaredMeanGradient[y]) this.squaredMeanGradient[y] = {};
                this.squaredMeanGradient[y][i] = nj.zeros(X.tolist().length);
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
                try {
                    const cost = await this.costFun(X, nj.array(this.w[label][proto]));
                    if (parseInt(label) === parseInt(y) && (highestCostCorr === undefined || cost > highestCostCorr)) {
                        // we found new best cost
                        highestCostCorr = cost;
                        labelCorr = label;
                        protoCorr = proto;
                    } else if (highestCostIncorr === undefined || cost > highestCostIncorr) {
                        // label differs, we need best incorrect to reject it
                        highestCostIncorr = cost;
                        labelIncorr = label;
                        protoIncorr = proto;
                    } // endElseIf
                } catch (e) {
                    throw `Error in optimize, costFun with x: ${X}, w: ${this.w[label][proto]}: ${e}`;
                } // endCatch
            } // endFor
        } // endFor

        // if nearest proto has not correct label, we have to learn
        if (highestCostCorr < highestCostIncorr) {
            await this.updatePrototype(y, labelIncorr, X, protoIncorr);
            await this.updatePrototype(y, labelCorr, X, protoCorr);
        } // endIf
    } // endOptimize

    async updateProtoypeAdamax(y, label, x, protoIdx) {
        // TODO
    } // endUpdatePrototypeAdamax

    async updatePrototypeAdadelta(y, label, x, protoIdx) {
        const dist = nj.array(x).subtract(this.w[label][protoIdx]);

        let gradient;
        const posterior = await this.calcPosterior(protoIdx, x, y, false);
        const posteriorWrtY = await this.calcPosterior(protoIdx, x, y, true);

        // now broadcast pos
        const broadcastedPosterior = [];
        const broadcastedPosteriorY = [];

        for (let i = 0; i < nj.array(this.w[y][protoIdx]).size; i++) {
            broadcastedPosterior[i] = posterior;
            broadcastedPosteriorY[i] = posteriorWrtY;
        } // endFor

        if (y === label) {
            gradient = nj.array(broadcastedPosteriorY).subtract(nj.array(broadcastedPosterior)).multiply(dist);
        } else {
            gradient = nj.negative(nj.array(broadcastedPosterior).multiply(dist));
        } // endElse

        const squaredGradient = gradient.pow(2);

        // we need to broadcast gamma
        let broadcastedGamma = [];
        let broadcastedResGamma = [];

        for (const i in squaredGradient.tolist()) {
            broadcastedGamma[i] = this.gamma;
            broadcastedResGamma[i] = 1 - this.gamma;
        } // endFor

        broadcastedGamma = nj.array(broadcastedGamma);
        broadcastedResGamma = nj.array(broadcastedResGamma);

        this.squaredMeanGradient[label][protoIdx] = broadcastedGamma.multiply(nj.array(this.squaredMeanGradient[label][protoIdx])).add(broadcastedResGamma.multiply(squaredGradient));

        // now we have to broadcast epsilon
        let broadcastedEpsilon = [];

        for (const i in squaredGradient.tolist()) {
            broadcastedEpsilon[i] = this.epsilon;
        } // endFor

        broadcastedEpsilon = nj.array(broadcastedEpsilon);
        const step = nj.sqrt(nj.array(this.squaredMeanStep[label][protoIdx]).add(broadcastedEpsilon).divide(nj.array(this.squaredMeanGradient[label][protoIdx]).add(broadcastedEpsilon))).multiply(gradient);
        this.squaredMeanStep[label][protoIdx] = broadcastedGamma.multiply(nj.array(this.squaredMeanStep[label][protoIdx])).add(broadcastedResGamma.multiply(step.pow(2)));

        // finally update the prototype
        this.logger.warn(`old proto: ${this.w[label][protoIdx]}`);

        this.w[label][protoIdx] = nj.array(this.w[label][protoIdx]).add(step);

        this.logger.warn(`new proto: ${this.w[label][protoIdx]}`);
    } // endUpdatePrototypeAdadelta

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
        const cosFun = await this.costFun(x, nj.array(this.w[y][protoIdx])) - fsMax;

        return Math.exp(cosFun) / sSum;

    } // endCalcPosterior

} // endClassRSLVQ

module.exports = RSLVQ;