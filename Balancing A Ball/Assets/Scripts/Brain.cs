using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEditor.PackageManager.Requests;
using System.Xml.Schema;
using System.Runtime.InteropServices.WindowsRuntime;

public class Replay
{
    public List<double> states;
    public double reward;
    public Replay(double xr,double ballz,double ballvx,double r)
    {
        states = new List<double>();
        states.Add(xr);
        states.Add(ballz);
        states.Add(ballvx);
        reward = r;
    }
}

public class Brain : MonoBehaviour
{
    public GameObject ball;
    ANN ann;
    float reward;
    List<Replay> replaymemory = new List<Replay>();
    int mcapacity = 10000;
    float discount = 0.99f;
    float explorerate = 100.0f;
    float maxeplorerate = 100.0f;
    float minexplorerate = 0.01f;
    float exploredecay = 0.0001f;
    Vector3 ballstartpos;
    int fallcount = 0;
    float tiltspeed = 0.5f;
    float timer = 0;
    float maxBalanceTime = 0;

    // Start is called before the first frame update
    void Start()
    {
        ann = new ANN(3, 2, 1, 6, 0.2f);
        ballstartpos = ball.transform.position;
        Time.timeScale = 5.0f;
    }

    GUIStyle guistyle = new GUIStyle();
    private void OnGUI()
    {
        guistyle.fontSize = 25;
        guistyle.normal.textColor = Color.white;
        GUI.BeginGroup(new Rect(10, 10, 600, 150));
        GUI.Box(new Rect(0, 0, 140, 140), "Stats : ", guistyle);
        GUI.Label(new Rect(10, 25, 500, 30), "Fails : " + fallcount, guistyle);
        GUI.Label(new Rect(10, 50, 500, 30), "DecayRate : " + explorerate, guistyle);
        GUI.Label(new Rect(10, 75, 500, 30), "Last Best Balance : " + maxBalanceTime, guistyle);
        GUI.Label(new Rect(10, 100, 500, 30), "This Balance : " + timer, guistyle);
        GUI.EndGroup();
    }
    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space))
        {
            Reset();
        }
    }
    private void FixedUpdate()
    {
        timer += Time.deltaTime;
        List<double> states = new List<double>();
        List<double> qs = new List<double>();

        states.Add(this.transform.rotation.x);
        states.Add(ball.transform.position.z);
        states.Add(ball.GetComponent<Rigidbody>().angularVelocity.x);

        qs = SoftMax(ann.CalcOutput(states));
        double maxQ = qs.Max();
        int maxQindex = qs.ToList().IndexOf(maxQ);
        explorerate = Mathf.Clamp(explorerate - exploredecay, minexplorerate, maxeplorerate);
        /*NO NEED OF EXPLORING IN THIS CASE AS ENVIRONMENT IS REALLY VERY SMALL
         *if(Random.Range(0,100)<explorerate)
         {
             maxQindex = Random.Range(0, 2);
         }*/
        if (maxQindex == 0)
        {
            this.transform.Rotate(Vector3.right, tiltspeed * (float)qs[maxQindex]);
        }
        else if (maxQindex == 1)
        {
            this.transform.Rotate(Vector3.right, -tiltspeed * (float)qs[maxQindex]);
        }
        if (ball.GetComponent<BallState>().dropped)
        {
            reward = -1;
        }
        else
        {
            reward = 0.1f;
        }
        Replay lastmemory = new Replay(this.transform.rotation.x, ball.transform.position.z, ball.GetComponent<Rigidbody>().angularVelocity.x, reward);
        if(replaymemory.Count>mcapacity)
        {
            replaymemory.RemoveAt(0);
        }
        replaymemory.Add(lastmemory);

        //Training and QLEARNING

        if(ball.GetComponent<BallState>().dropped)
        {
            for(int i=replaymemory.Count-1;i>=0;i--)
            {
                List<double> toutputs_old = new List<double>();
                List<double> toutputs_new = new List<double>();
                toutputs_old = SoftMax(ann.CalcOutput(replaymemory[i].states));
                double maxQ_old = toutputs_old.Max();
                int action = toutputs_old.ToList().IndexOf(maxQ_old);

                double feedback;
                if(i==replaymemory.Count-1 || replaymemory[i].reward==-1)
                {
                    feedback = replaymemory[i].reward;
                }
                else
                {
                    toutputs_new = SoftMax(ann.CalcOutput(replaymemory[i + 1].states));
                    maxQ = toutputs_new.ToList().Max();
                    feedback = (replaymemory[i].reward + discount * maxQ);  //BELLMAN EQUATION
                }
                toutputs_old[action] = feedback;
                ann.Train(replaymemory[i].states, toutputs_old);
            }
            if(timer>maxBalanceTime)
            {
                maxBalanceTime = timer;
            }
            timer = 0;
            ball.GetComponent<BallState>().dropped = false;
            this.transform.rotation = Quaternion.identity;
            Reset();
            replaymemory.Clear();
            fallcount++;
        }
    }
    private void Reset()
    {
        ball.transform.position = ballstartpos;
        ball.GetComponent<Rigidbody>().velocity = new Vector3(0, 0, 0);
        ball.GetComponent<Rigidbody>().angularVelocity = new Vector3(0, 0, 0);
    }
    List<double> SoftMax(List<double>values)
    {
        double max = values.Max();
        float scale = 0.0f;
        for(int i=0;i<values.Count;++i)
        {
            scale += Mathf.Exp((float)(values[i] - max));
        }
        List<double> result = new List<double>();
        for(int i=0;i<values.Count;++i)
        {
            result.Add(Mathf.Exp((float)(values[i] - max)) / scale);
        }
        return result;
    }
}
