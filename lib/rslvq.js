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

        // if we already have prototypes init them
        if (this.initialPrototypes) {
            this.validatePrototypes(null, null);
        } // endIf
    } // endConstructor

    async partialFit(X, y) {
        // validate prototypes
        await this.validatePrototypes(X, y);

        // optimize by adadelta
        await this.optimize();
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

    } // endOptimize

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
        let d = x.subtract(w);
        d = d.T.dot(d);
        return d.divide(2 * this.sigma).multiply(-1);
    } // endCostFun

} // endClassRSLVQ

module.exports = RSLVQ;