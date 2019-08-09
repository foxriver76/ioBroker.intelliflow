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
        this.initialFit = true;
        this.protosInitialized = {};
        this.w = {};

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
} // endClassRSLVQ

module.exports = RSLVQ;